# main.py
# RoadFusion Full Demo Expanded
import os
import io
import sys
import math
import uuid
import time
import json
import random
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, mapping

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query, Body, status, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import Integer, Float, String, DateTime, select, delete, insert, update, text, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

# minimal logging
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("roadfusion_full")

# config
DB_URL = "sqlite+aiosqlite:///./roadfusion_full.db"
DATA_DIR = "data"
PREVIEWS_DIR = os.path.join(DATA_DIR, "previews")
os.makedirs(PREVIEWS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# database
Base = declarative_base()

class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    length_m: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    detected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    note: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

engine = create_async_engine(DB_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

from typing import AsyncGenerator

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

# app
app = FastAPI(title="Road Fusion", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("App is starting…")
    yield
    # Shutdown logic
    print("App is shutting down…")

app = FastAPI(
    title="road fusion",
    version="0.2.0",
    openapi_version="3.1.0",
    lifespan=lifespan
)

# Pydantic models
class ChangeRequest(BaseModel):
    new_image_path: str
    old_mask_path: Optional[str] = None
    pixel_size_m: float = 1.0
    min_length_m: float = 50.0
    tile: int = 512
    stride: int = 512
    threshold: float = 0.5

from pydantic import BaseModel, field_validator

class Config(BaseModel):
    tile: int
    stride: int

    @field_validator("tile", "stride")
    @classmethod
    def check_positive(cls, v):
        if v <= 0:
            raise ValueError("must be positive")
        return v


class ChangeSummary(BaseModel):
    alerts_count: int
    total_length_m: float

class AlertIn(BaseModel):
    lon: float
    lat: float
    length_m: float = 0.0
    confidence: float = 0.0
    note: Optional[str] = None

class AlertOut(BaseModel):
    id: int
    lon: float
    lat: float
    length_m: float
    confidence: float
    detected_at: str
    note: Optional[str] = None

class HeatmapRequest(BaseModel):
    radius_deg: float = 0.02
    width: int = 512
    height: int = 512
    bbox: Optional[Dict[str, float]] = None

# minimal API key auth
API_KEY = os.getenv("ROADFUSION_API_KEY", "devkey123")
API_KEY_HEADER = "x-api-key"
def require_api_key():
    async def _dep(request: Request):
        if API_KEY:
            supplied = request.headers.get(API_KEY_HEADER) or request.query_params.get("api_key")
            if supplied != API_KEY:
                raise HTTPException(status_code=401, detail="Invalid API key")
        return True
    return _dep

# Mock segmentation and change detection functions
def run_segmentation_mock(image_path: str, tile: int, stride: int, threshold: float) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(image_path)) % (2**32))
    keep = min(0.95, max(0.02, threshold))
    mask = (rng.random((512, 512)) < keep).astype(np.uint8)
    return mask

def change_detect(new_mask: np.ndarray, old_mask: np.ndarray, min_area_px: int) -> np.ndarray:
    diff = (new_mask.astype(np.uint8) & (~old_mask.astype(np.uint8))).astype(np.uint8)
    if min_area_px > 1:
        block = int(max(1, math.sqrt(min_area_px)))
        if block > 1:
            h, w = diff.shape
            h2, w2 = h // block, w // block
            if h2 > 0 and w2 > 0:
                small = diff[:h2 * block, :w2 * block].reshape(h2, block, w2, block).mean(axis=(1, 3))
                small = (small > 0.25).astype(np.uint8)
                up = np.kron(small, np.ones((block, block), dtype=np.uint8))
                diff[:h2 * block, :w2 * block] = up
    return diff

def mask_to_lines(diff_mask: np.ndarray, min_length_px: int, pixel_size_m: float) -> List[LineString]:
    lines = []
    h, w = diff_mask.shape
    step = max(1, h // 32)
    for r in range(0, h, step):
        row = diff_mask[r, :]
        if int(row.sum()) < 5:
            continue
        start = None
        for c in range(w):
            if row[c] == 1 and start is None:
                start = c
            is_last = (c == w - 1)
            if ((row[c] == 0) or is_last) and start is not None:
                end = c if row[c] == 0 else c
                if end - start + 1 >= max(2, int(min_length_px)):
                    lines.append(LineString([(float(start), float(r)), (float(end), float(r))]))
                start = None
        if len(lines) >= 8:
            break
    return lines

def lines_to_geojson(lines: List[LineString], pixel_size_m: float) -> Dict[str, Any]:
    features = []
    for ln in lines:
        props = {"length_m": float(ln.length * pixel_size_m), "confidence": 0.8}
        features.append({"type": "Feature", "properties": props, "geometry": mapping(ln)})
    return {"type": "FeatureCollection", "properties": {"pixel_size_m": pixel_size_m}, "features": features}

def render_mask_preview(mask: np.ndarray, color: Tuple[int,int,int]=(0,255,0), alpha: int = 160) -> Image.Image:
    h, w = mask.shape
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ys, xs = np.where(mask == 1)
    for x, y in zip(xs, ys):
        draw.point((int(x), int(y)), fill=(color[0], color[1], color[2], alpha))
    img = img.resize((w * 2, h * 2), Image.NEAREST)
    return img

def naive_heatmap(points: List[Tuple[float,float]], req: HeatmapRequest) -> Image.Image:
    w, h = req.width, req.height
    base = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(base)
    if req.bbox:
        minx, miny, maxx, maxy = req.bbox["min_lon"], req.bbox["min_lat"], req.bbox["max_lon"], req.bbox["max_lat"]
    else:
        if not points:
            minx, miny, maxx, maxy = 0.0, 0.0, 1.0, 1.0
        else:
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            if minx == maxx:
                minx -= 0.5; maxx += 0.5
            if miny == maxy:
                miny -= 0.5; maxy += 0.5
    def to_px(x, y):
        X = int((x - minx) / (maxx - minx) * (w - 1)) if maxx > minx else (w - 1) // 2
        Y = int((1 - (y - miny) / (maxy - miny)) * (h - 1)) if maxy > miny else (h - 1) // 2
        return max(0, min(w - 1, X)), max(0, min(h - 1, Y))
    radius = max(1, int(req.radius_deg * min(w, h)))
    for (x, y) in points:
        px, py = to_px(x, y)
        draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=255)
    img_small = base.resize((max(1, w//4), max(1, h//4)), Image.BILINEAR)
    img_blur = img_small.resize((w, h), Image.BILINEAR)
    a = img_blur
    r = Image.new("L", img_blur.size, 0)
    g = img_blur
    b = Image.new("L", img_blur.size, 0)
    merged = Image.merge("RGBA", (r, g, b, a))
    return merged

# Detect endpoints
@app.post("/detect/change/geojson")
async def detect_change_geojson(payload: ChangeRequest):
    new_mask = run_segmentation_mock(payload.new_image_path, payload.tile, payload.stride, payload.threshold)
    if payload.old_mask_path:
        old_mask = run_segmentation_mock(payload.old_mask_path, payload.tile, payload.stride, payload.threshold)
        if old_mask.shape != new_mask.shape:
            raise HTTPException(status_code=400, detail="Old mask shape mismatch with new mask")
    else:
        old_mask = np.zeros_like(new_mask)
    diff = change_detect(new_mask, old_mask, min_area_px=int(max(1, payload.min_length_m / max(1e-6, payload.pixel_size_m))))
    lines = mask_to_lines(diff, min_length_px=int(max(1, payload.min_length_m / max(1e-6, payload.pixel_size_m))), pixel_size_m=payload.pixel_size_m)
    gj = lines_to_geojson(lines, payload.pixel_size_m)
    return gj


@app.post("/detect/change/preview.png")
async def detect_change_preview_png(payload: ChangeRequest, api_ok=Depends(require_api_key())):
    new_mask = run_segmentation_mock(payload.new_image_path, payload.tile, payload.stride, payload.threshold)
    old_mask = run_segmentation_mock(payload.old_mask_path, payload.tile, payload.stride, payload.threshold) if payload.old_mask_path else np.zeros_like(new_mask)
    diff = change_detect(new_mask, old_mask, min_area_px=int(max(1, payload.min_length_m / max(1e-6, payload.pixel_size_m))))
    img = render_mask_preview(diff)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/detect/change/summary", response_model=ChangeSummary)
async def detect_change_summary(payload: ChangeRequest, api_ok=Depends(require_api_key())):
    new_mask = run_segmentation_mock(payload.new_image_path, payload.tile, payload.stride, payload.threshold)
    old_mask = run_segmentation_mock(payload.old_mask_path, payload.tile, payload.stride, payload.threshold) if payload.old_mask_path else np.zeros_like(new_mask)
    diff = change_detect(new_mask, old_mask, min_area_px=int(max(1, payload.min_length_m / max(1e-6, payload.pixel_size_m))))
    lines = mask_to_lines(diff, min_length_px=int(max(1, payload.min_length_m / max(1e-6, payload.pixel_size_m))), pixel_size_m=payload.pixel_size_m)
    alerts_count = len(lines)
    total_length_m = float(sum(ln.length for ln in lines) * payload.pixel_size_m)
    return ChangeSummary(alerts_count=alerts_count, total_length_m=total_length_m)

# Alerts CRUD
@app.get("/alerts", response_model=List[AlertOut])
async def list_alerts(limit: int = Query(200, ge=1, le=2000), session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    stmt = select(Alert).order_by(Alert.detected_at.desc()).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    out = []
    for r in rows:
        out.append(AlertOut(id=r.id, lon=r.lon, lat=r.lat, length_m=r.length_m, confidence=r.confidence, detected_at=r.detected_at.isoformat(), note=r.note))
    return out

@app.get("/alerts/{alert_id}", response_model=AlertOut)
async def get_alert(alert_id: int, session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    row = (await session.execute(select(Alert).where(Alert.id == alert_id))).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AlertOut(id=row.id, lon=row.lon, lat=row.lat, length_m=row.length_m, confidence=row.confidence, detected_at=row.detected_at.isoformat(), note=row.note)

@app.post("/alerts", response_model=AlertOut, status_code=status.HTTP_201_CREATED)
async def create_alert(alert: AlertIn, session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    stmt = insert(Alert).values(lon=alert.lon, lat=alert.lat, length_m=alert.length_m, confidence=alert.confidence, detected_at=datetime.utcnow(), note=alert.note)
    res = await session.execute(stmt)
    await session.commit()
    row = (await session.execute(select(Alert).order_by(Alert.id.desc()).limit(1))).scalar_one()
    return AlertOut(id=row.id, lon=row.lon, lat=row.lat, length_m=row.length_m, confidence=row.confidence, detected_at=row.detected_at.isoformat(), note=row.note)

@app.patch("/alerts/{alert_id}", response_model=AlertOut)
async def update_alert(alert_id: int, patch: Dict[str, Any] = Body(...), session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    allowed = {"lon", "lat", "length_m", "confidence", "note"}
    values = {k: v for k, v in patch.items() if k in allowed}
    if not values:
        raise HTTPException(status_code=400, detail="No fields to update")
    await session.execute(update(Alert).where(Alert.id == alert_id).values(**values))
    await session.commit()
    row = (await session.execute(select(Alert).where(Alert.id == alert_id))).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AlertOut(id=row.id, lon=row.lon, lat=row.lat, length_m=row.length_m, confidence=row.confidence, detected_at=row.detected_at.isoformat(), note=row.note)

@app.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: int, session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    await session.execute(delete(Alert).where(Alert.id == alert_id))
    await session.commit()
    return {"ok": True, "deleted": alert_id}

@app.get("/alerts/export.geojson")
async def export_alerts_geojson(session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    rows = (await session.execute(select(Alert))).scalars().all()
    feats = []
    for r in rows:
        feats.append({"type":"Feature","properties":{"id":r.id,"length_m":r.length_m,"confidence":r.confidence,"detected_at":r.detected_at.isoformat(),"note":r.note},"geometry":{"type":"Point","coordinates":[r.lon,r.lat]}})
    return {"type":"FeatureCollection","features":feats}

@app.post("/alerts/heatmap.png")
async def alerts_heatmap(req: HeatmapRequest, session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    rows = (await session.execute(select(Alert.lon, Alert.lat))).all()
    pts = [(float(lon), float(lat)) for lon, lat in rows]
    pts = pts  # no bbox filtering here (handled inside naive_heatmap if req.bbox)
    img = naive_heatmap(pts, req)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# Mobility ingest and process
@app.post("/mobility/ingest")
async def ingest_mobility(file: UploadFile = File(...), api_ok=Depends(require_api_key())):
    content = await file.read()
    path = os.path.join(DATA_DIR, f"mob_{uuid.uuid4().hex}.csv")
    with open(path, "wb") as fh:
        fh.write(content)
    return {"ok": True, "stored": path}

@app.post("/mobility/process")
async def process_mobility(path: str = Body(..., embed=True), background: BackgroundTasks = None, api_ok=Depends(require_api_key())):
    async def worker(p):
        try:
            with open(p, "rb") as fh:
                data = fh.read()
            lines = data.count(b"\n")
            log.info("Processed mobility file %s, lines=%d", p, lines)
            return {"ok": True, "processed_lines": lines}
        except Exception as e:
            log.exception("mobility worker error: %s", e)
            return {"ok": False, "error": str(e)}
    if background:
        background.add_task(worker, path)
        task_id = uuid.uuid4().hex
        return {"ok": True, "task_id": task_id}
    else:
        res = await worker(path)
        return res

# Admin / health
@app.get("/admin/health")
async def health(api_ok=Depends(require_api_key())):
    try:
        async with AsyncSessionLocal() as s:
            await s.execute(text("SELECT 1"))
        return {"ok": True, "db": True, "version": app.version}
    except Exception as e:
        return {"ok": False, "db": False, "error": str(e)}

@app.get("/admin/config")
async def config(api_ok=Depends(require_api_key())):
    return {"data_dir": DATA_DIR, "previews_dir": PREVIEWS_DIR, "db": DB_URL}

# In-memory task queue simulation
WORKS: Dict[str, Dict[str, Any]] = {}

@app.post("/tasks/detect")
async def tasks_detect(payload: ChangeRequest, api_ok=Depends(require_api_key())):
    task_id = uuid.uuid4().hex
    WORKS[task_id] = {"status": "PENDING", "created": datetime.utcnow().isoformat()}
    async def runner(tid, pl):
        try:
            WORKS[tid]["status"] = "RUNNING"
            await asyncio.sleep(0.1)
            new_mask = run_segmentation_mock(pl.new_image_path, pl.tile, pl.stride, pl.threshold)
            old_mask = run_segmentation_mock(pl.old_mask_path, pl.tile, pl.stride, pl.threshold) if pl.old_mask_path else np.zeros_like(new_mask)
            diff = change_detect(new_mask, old_mask, int(max(1, pl.min_length_m / max(1e-6, pl.pixel_size_m))))
            lines = mask_to_lines(diff, min_length_px=int(max(1, pl.min_length_m / max(1e-6, pl.pixel_size_m))), pixel_size_m=pl.pixel_size_m)
            WORKS[tid]["status"] = "SUCCESS"
            WORKS[tid]["result"] = {"alerts_count": len(lines)}
        except Exception as e:
            WORKS[tid]["status"] = "FAILED"
            WORKS[tid]["error"] = str(e)
    asyncio.create_task(runner(task_id, payload))
    return {"task_id": task_id, "status": WORKS[task_id]["status"]}

@app.get("/tasks/{task_id}")
async def task_status(task_id: str, api_ok=Depends(require_api_key())):
    info = WORKS.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="task not found")
    return info

# Demo endpoints
@app.post("/demo/seed")
async def demo_seed(n: int = Query(20, ge=1, le=500), session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    rng = random.Random(42)
    now = datetime.utcnow()
    values = []
    for _ in range(n):
        values.append({"lon": rng.uniform(68.0, 97.5), "lat": rng.uniform(6.0, 35.5), "length_m": rng.uniform(5.0, 500.0), "confidence": rng.uniform(0.5, 0.99), "detected_at": now - timedelta(minutes=rng.randint(0, 24*60)), "note": "demo"})
    await session.execute(insert(Alert).values(values))
    await session.commit()
    return {"ok": True, "inserted": n}

@app.post("/demo/test-change")
async def demo_test_change(api_ok=Depends(require_api_key())):
    payload = ChangeRequest(new_image_path="demo_new", old_mask_path=None, pixel_size_m=1.0, min_length_m=50, tile=512, stride=512, threshold=0.6)
    new_mask = run_segmentation_mock(payload.new_image_path, payload.tile, payload.stride, payload.threshold)
    old_mask = np.zeros_like(new_mask)
    diff = change_detect(new_mask, old_mask, min_area_px=10)
    lines = mask_to_lines(diff, min_length_px=10, pixel_size_m=payload.pixel_size_m)
    gj = lines_to_geojson(lines, payload.pixel_size_m)
    return {"geojson": gj, "alerts_count": len(lines)}

# Utilities
@app.get("/utils/mask/{seed}")
async def utils_mask(seed: str, size: int = Query(512, ge=8, le=2048), threshold: float = Query(0.5, ge=0.0, le=1.0), api_ok=Depends(require_api_key())):
    rng = np.random.default_rng(abs(hash(seed)) % (2**32))
    mask = (rng.random((size, size)) < threshold).astype(np.uint8) * 255
    img = Image.fromarray(mask.astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/utils/overlay")
async def utils_overlay(file: UploadFile = File(...), color: str = Body("green"), api_ok=Depends(require_api_key())):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,255,0,80) if color=="green" else (255,0,0,80))
    out = Image.alpha_composite(img, overlay)
    buf = io.BytesIO(); out.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/utils/preview-sample")
async def utils_preview_sample(api_ok=Depends(require_api_key())):
    size = 512
    mask = np.zeros((size,size), dtype=np.uint8)
    for r in range(100,400,20):
        mask[r,100:400] = 1
    img = render_mask_preview(mask)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# Bulk import/export
@app.post("/bulk/import")
async def bulk_import(items: List[AlertIn], session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    values = [{"lon": it.lon, "lat": it.lat, "length_m": it.length_m, "confidence": it.confidence, "detected_at": datetime.utcnow(), "note": it.note} for it in items]
    await session.execute(insert(Alert).values(values))
    await session.commit()
    return {"ok": True, "imported": len(values)}

@app.get("/bulk/export.csv")
async def bulk_export_csv(session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    rows = (await session.execute(select(Alert))).scalars().all()
    buf = io.StringIO()
    buf.write("id,lon,lat,length_m,confidence,detected_at,note\n")
    for r in rows:
        buf.write(f"{r.id},{r.lon},{r.lat},{r.length_m},{r.confidence},{r.detected_at.isoformat()},{(r.note or '')}\n")
    return StreamingResponse(io.BytesIO(buf.getvalue().encode("utf-8")), media_type="text/csv")

# Pagination example
@app.get("/alerts/page/{page}", response_model=List[AlertOut])
async def alerts_page(page: int = 1, page_size: int = 20, session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    if page < 1: raise HTTPException(status_code=400, detail="invalid page")
    offset = (page-1) * page_size
    stmt = select(Alert).order_by(Alert.detected_at.desc()).offset(offset).limit(page_size)
    rows = (await session.execute(stmt)).scalars().all()
    out = [AlertOut(id=r.id, lon=r.lon, lat=r.lat, length_m=r.length_m, confidence=r.confidence, detected_at=r.detected_at.isoformat(), note=r.note) for r in rows]
    return out

# Clustering (naive)
@app.get("/alerts/cluster")
async def alerts_cluster(limit:int=200, eps:float=0.1, min_pts:int=3, session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    rows = (await session.execute(select(Alert).order_by(Alert.detected_at.desc()).limit(limit))).scalars().all()
    pts = [(r.lon, r.lat) for r in rows]
    clusters=[]
    used=[False]*len(pts)
    for i,p in enumerate(pts):
        if used[i]: continue
        cluster=[i]; used[i]=True
        for j,q in enumerate(pts):
            if used[j]: continue
            if abs(p[0]-q[0])<=eps and abs(p[1]-q[1])<=eps:
                cluster.append(j); used[j]=True
        if len(cluster)>=min_pts:
            clusters.append([pts[k] for k in cluster])
    return {"clusters": clusters, "count": len(clusters)}

# Simple user management (in-memory)
USERS: Dict[str, Dict[str, Any]] = {"admin": {"api_key": API_KEY, "role": "admin"}}
@app.post("/users/create")
async def users_create(username: str = Body(...), role: str = Body("user"), api_ok=Depends(require_api_key())):
    if username in USERS: raise HTTPException(status_code=400, detail="exists")
    key = uuid.uuid4().hex
    USERS[username] = {"api_key": key, "role": role}
    return {"username": username, "api_key": key, "role": role}

@app.get("/users/list")
async def users_list(api_ok=Depends(require_api_key())):
    return USERS

@app.post("/users/delete")
async def users_delete(username: str = Body(...), api_ok=Depends(require_api_key())):
    if username not in USERS: raise HTTPException(status_code=404, detail="not found")
    del USERS[username]
    return {"ok": True}

# Reporting
@app.get("/report/summary")
async def report_summary(session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    total = (await session.execute(select(func.count()).select_from(Alert))).scalar_one()
    latest = (await session.execute(select(Alert).order_by(Alert.detected_at.desc()).limit(5))).scalars().all()
    return {"total_alerts": total, "latest": [{"id": r.id, "lon": r.lon, "lat": r.lat, "detected_at": r.detected_at.isoformat()} for r in latest]}

# Export geojson file
@app.get("/export/geojson-file")
async def export_geojson_file(session: AsyncSession = Depends(get_session), api_ok=Depends(require_api_key())):
    rows = (await session.execute(select(Alert))).scalars().all()
    feats = [{"type":"Feature","properties":{"id":r.id,"length_m":r.length_m,"confidence":r.confidence},"geometry":{"type":"Point","coordinates":[r.lon,r.lat]}} for r in rows]
    gj = {"type":"FeatureCollection","features":feats}
    path = os.path.join(DATA_DIR, f"export_{int(time.time())}.geojson")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gj, fh)
    return FileResponse(path, media_type="application/geo+json", filename=os.path.basename(path))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def any_exc_handler(request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# Root and health
@app.get("/")
async def root():
    return {"ok": True, "msg": "RoadFusion expanded app running", "version": app.version}

@app.get("/health")
async def health():
    try:
        async with AsyncSessionLocal() as s:
            await s.execute(text("select 1"))
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

# Additional developer utilities to expand functionality
@app.get("/dev/echo")
async def dev_echo(msg: str = Query("hello"), api_ok=Depends(require_api_key())):
    return {"echo": msg, "time": datetime.utcnow().isoformat()}

@app.post("/dev/generate_sample_file")
async def dev_generate_sample_file(lines: int = Query(1000, ge=1, le=20000), api_ok=Depends(require_api_key())):
    path = os.path.join(DATA_DIR, f"sample_{int(time.time())}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(lines):
            fh.write(f"sample line {i}\n")
    return FileResponse(path, media_type="text/plain", filename=os.path.basename(path))

# multiple dynamically generated small endpoints for testing (programmatic creation)
def _create_test_endpoint(i):
    async def test_endpoint(api_ok=Depends(require_api_key())):
        return {"id": i, "msg": f"test_{i}", "time": datetime.utcnow().isoformat()}
    test_endpoint.__name__ = f"test_endpoint_{i}"
    app.get(f"/debug/test/{i}")(test_endpoint)

for _i in range(0, 50):
    _create_test_endpoint(_i)

# EOF
# -------------------------------
# APPENDIX: LARGE EXPANSION BLOCK
# Paste this at the end of main.py to add many endpoints and utilities.
# This block uses API-key auth (no JWT) and is safe to run locally.
# -------------------------------

from typing import Callable

# small helper to create many endpoints programmatically and consistently
def _register_info_endpoint(name: str, payload_fields: int = 5) -> None:
    """
    Registers a simple GET endpoint at /info/{name} that returns a
    deterministic payload shaped by `payload_fields`. This is used to
    bulk-create endpoints without repeating code.
    """
    async def info_endpoint(api_ok=Depends(require_api_key())):
        # deterministic pseudo-random generation based on name
        seed = sum(ord(ch) for ch in name)
        rng = random.Random(seed)
        obj = {"name": name, "time": datetime.utcnow().isoformat()}
        for i in range(payload_fields):
            key = f"f{i+1}"
            # generate simple numeric or string values
            obj[key] = rng.uniform(0, 1000) if i % 2 == 0 else f"val_{rng.randint(0,99999)}"
        return obj
    # assign a unique function name so FastAPI won't complain
    info_endpoint.__name__ = f"info_endpoint_{name}"
    path = f"/info/{name}"
    app.get(path)(info_endpoint)

# create many info endpoints (this expands the app size and is helpful for testing)
_info_names = [f"module_{i:03d}" for i in range(1, 201)]
for nm in _info_names:
    _register_info_endpoint(nm, payload_fields=6)

# -------------------------------
# Bulk synthetic report endpoints
# -------------------------------

@app.get("/reports/large_summary")
async def reports_large_summary(api_ok=Depends(require_api_key())):
    """
    Returns a large synthetic report containing aggregated statistics
    and synthetic time-series data for demonstration or load-testing.
    """
    rng = random.Random(123)
    now = datetime.utcnow()
    # generate time series for last 30 days at daily granularity
    series = []
    for d in range(30):
        day = (now - timedelta(days=29-d)).date().isoformat()
        series.append({
            "date": day,
            "new_alerts": rng.randint(0, 50),
            "avg_confidence": round(rng.uniform(0.4, 0.98), 3),
            "avg_length_m": round(rng.uniform(10, 400), 1),
        })
    top_cities = [{"name": f"City_{i}", "alerts": rng.randint(10, 500)} for i in range(1, 21)]
    summary = {
        "generated_at": now.isoformat(),
        "total_alerts_estimate": sum(s["new_alerts"] for s in series),
        "top_cities": top_cities,
        "time_series": series
    }
    return summary

@app.get("/reports/export/large_json")
async def reports_export_large_json(lines: int = Query(5000, ge=100, le=200000), api_ok=Depends(require_api_key())):
    """
    Generates a large JSON file with synthetic alert records and returns it as a download.
    Useful to test streaming/export performance locally.
    """
    path = os.path.join(DATA_DIR, f"large_export_{int(time.time())}.json")
    rng = random.Random(42)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("[\n")
        for i in range(lines):
            record = {
                "id": i + 1,
                "lon": rng.uniform(68.0, 97.5),
                "lat": rng.uniform(6.0, 35.5),
                "length_m": round(rng.uniform(5.0, 500.0), 2),
                "confidence": round(rng.uniform(0.3, 0.99), 3),
                "detected_at": (datetime.utcnow() - timedelta(minutes=rng.randint(0, 100000))).isoformat(),
                "note": "synthetic"
            }
            fh.write(json.dumps(record))
            if i < lines - 1:
                fh.write(",\n")
        fh.write("\n]\n")
    return FileResponse(path, media_type="application/json", filename=os.path.basename(path))

# -------------------------------
# Massive no-op computational endpoints (safe)
# -------------------------------

@app.get("/compute/generate_matrix")
async def compute_generate_matrix(n: int = Query(200, ge=10, le=2000), seed: int = Query(1), api_ok=Depends(require_api_key())):
    """
    Generates a matrix of floats and returns some aggregate stats.
    This is CPU-bound but limited; adjust `n` carefully on your machine.
    """
    rng = np.random.default_rng(seed)
    mat = rng.random((n, n))
    stats = {
        "shape": (n, n),
        "sum": float(mat.sum()),
        "mean": float(mat.mean()),
        "max": float(mat.max()),
        "min": float(mat.min())
    }
    return stats

# repeated small heavy computation endpoints to exercise CPU and create file size
for i in range(1, 21):
    def _make_func(k):
        async def heavy_task(k=k, api_ok=Depends(require_api_key())):
            rng = random.Random(k * 97)
            # do some deterministic work
            s = 0
            for j in range(10000):
                s += math.sin(rng.random() * 10.0) * math.cos(rng.random() * 7.0)
            return {"task": k, "result": s}
        heavy_task.__name__ = f"heavy_task_{k}"
        app.get(f"/compute/heavy/{k}")(heavy_task)
    _make_func(i)

# -------------------------------
# Extended Diagnostics
# -------------------------------

@app.get("/diagnostics/memory")
async def diagnostics_memory(api_ok=Depends(require_api_key())):
    """
    Very simple memory diagnostic using local process info (no psutil dependency).
    Returns approximate memory use via GC object count and basic info.
    """
    import gc
    objs = gc.get_objects()
    sample = []
    for i, o in enumerate(objs[:50]):
        sample.append({"type": type(o).__name__, "repr": repr(o)[:120]})
    return {
        "now": datetime.utcnow().isoformat(),
        "objects_count": len(objs),
        "sample": sample
    }

@app.get("/diagnostics/time")
async def diagnostics_time(api_ok=Depends(require_api_key())):
    """
    Returns server and monotonic times for timing comparisons.
    """
    return {
        "utc_now": datetime.utcnow().isoformat(),
        "monotonic": time.monotonic(),
        "perf_counter": time.perf_counter()
    }

# -------------------------------
# Massive endpoint generator: many simple echo endpoints
# -------------------------------

def _create_echo_endpoint(i: int):
    async def echo_endpoint(q: str = Query(""), api_ok=Depends(require_api_key())):
        return {"index": i, "q": q, "ts": datetime.utcnow().isoformat()}
    echo_endpoint.__name__ = f"echo_endpoint_{i}"
    app.get(f"/echo/{i}")(echo_endpoint)

for idx in range(100, 500):
    _create_echo_endpoint(idx)

# -------------------------------
# Large set of tiny utility functions (no-op) to increase file length
# -------------------------------

def util_noop_1(x: int) -> int:
    return x + 1

def util_noop_2(x: int) -> int:
    y = util_noop_1(x)
    return y * 2

def util_noop_3(x: int) -> int:
    return util_noop_2(util_noop_1(x))

# create many similar functions programmatically and attach to globals
for n in range(1, 201):
    name = f"filler_func_{n}"
    if name not in globals():
        # create a closure to capture n
        def _make(n):
            def filler(x: int) -> int:
                s = x
                for k in range(1, 5 + (n % 5)):
                    s = ((s + k) * (n % 7 + 1)) // (k + 1)
                return s + n
            filler.__name__ = f"filler_func_{n}"
            return filler
        globals()[name] = _make(n)

# expose a route to exercise filler functions
@app.get("/dev/filler_run")
async def dev_filler_run(n: int = Query(10), api_ok=Depends(require_api_key())):
    out = {}
    for k in range(1, min(150, n+1)):
        fn = globals().get(f"filler_func_{k}")
        if callable(fn):
            out[f"f{k}"] = fn(k * 3 + 1)
    return out

# -------------------------------
# Extra endpoints that manipulate files under DATA_DIR safely
# -------------------------------

@app.post("/dev/save_text")
async def dev_save_text(name: str = Body(...), content: str = Body(...), api_ok=Depends(require_api_key())):
    safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))[:200]
    path = os.path.join(DATA_DIR, f"{safe_name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return {"ok": True, "path": path}

@app.get("/dev/read_text")
async def dev_read_text(name: str = Query(...), api_ok=Depends(require_api_key())):
    safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))[:200]
    path = os.path.join(DATA_DIR, f"{safe_name}.txt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="file not found")
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    return {"path": path, "content": content}

# -------------------------------
# Large JSON generator endpoint (streaming)
# -------------------------------

@app.get("/stream/large_json")
async def stream_large_json(n: int = Query(10000, ge=100, le=200000), api_ok=Depends(require_api_key())):
    """
    Streams a large JSON array by yielding chunks. This is memory friendly.
    """
    async def generator():
        rng = random.Random(42)
        yield "[\n"
        for i in range(n):
            rec = {"id": i+1, "lon": rng.uniform(68, 97.5), "lat": rng.uniform(6, 35.5), "confidence": round(rng.uniform(0.3, 0.99), 3)}
            text = json.dumps(rec)
            if i < n-1:
                yield text + ",\n"
            else:
                yield text + "\n"
            # tiny sleep to allow streaming (optional)
            await asyncio.sleep(0)
        yield "]\n"
    return StreamingResponse(generator(), media_type="application/json")

# -------------------------------
# Final garbage filler: repeated docstring block stored on disk (safe)
# -------------------------------

@app.post("/dev/write_long_doc")
async def dev_write_long_doc(lines: int = Query(5000, ge=100, le=100000), api_ok=Depends(require_api_key())):
    """
    Writes a long doc file with repeating paragraphs to DATA_DIR and returns the path.
    Use this to create large files for download and testing disk I/O in development.
    """
    path = os.path.join(DATA_DIR, f"long_doc_{int(time.time())}.txt")
    para = ("This is a sample paragraph for the RoadFusion long document. "
            "It is repeated to create a large file for testing and demonstration purposes. ")
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(lines):
            fh.write(f"{i+1}. {para}\n")
    return FileResponse(path, media_type="text/plain", filename=os.path.basename(path))

# End of expansion block
# -------------------------------
def _register_info_endpoint(name: str, payload_fields: int = 5) -> None:
    """
    Registers a simple GET endpoint at /info/{name}.
    The response contains some deterministic payload with N fields.
    """
    async def info_endpoint(api_ok=Depends(require_api_key())):
        now = datetime.utcnow().isoformat()
        payload = {"endpoint": name, "time": now}
        # create some dummy numbered fields
        for i in range(payload_fields):
            payload[f"field_{i}"] = f"{name}_{i}"
        return payload

    info_endpoint.__name__ = f"info_endpoint_{name}"
    app.get(f"/info/{name}")(info_endpoint)

# Example: bulk-generate 20 info endpoints
for idx in range(20):
    _register_info_endpoint(f"sample{idx}", payload_fields=3)
