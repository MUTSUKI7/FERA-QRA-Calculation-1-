# -*- coding: utf-8 -*-
"""
QRA Risk Contour Suite (통합 앱)
- 위성사진 + QRA 엑셀 파일을 사이드바에서 한 번만 업로드
- st.tabs로 3가지 결과(Explosion / Thermal / LSIR)를 한 화면에서 확인
- 각 탭은 필요한 시트가 있는지 개별적으로 검증 -> 없으면 그 탭만 경고, 나머지는 정상 작동
"""

import io
import math
import re
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="QRA Risk Contour Suite", layout="wide")
st.title("🛢️ QRA Risk Contour Suite")
st.caption("위성사진과 QRA 엑셀 파일을 한 번만 업로드하면 폭발 / 열복사 / LSIR 결과를 탭에서 각각 확인할 수 있습니다.")

# ------------------------------------------------------------------
# 공통 입력 (사이드바)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("공통 입력")
    image_file = st.file_uploader("위성사진 (png/jpg)", type=["png", "jpg", "jpeg"])
    excel_file = st.file_uploader("QRA 엑셀 파일 (xlsx)", type=["xlsx"])
    grid_extent_m = st.number_input("격자 전체 범위 (m)", min_value=1.0, value=1000.0, step=1.0)
    cell_size_m = st.number_input(
        "격자 한 칸 크기 (m)",
        min_value=0.1, value=50.0, step=1.0,
    )
    grid_cell_count = grid_extent_m / cell_size_m
    show_grid_lines = st.checkbox("이미지 위에 격자선 그리기", value=True)

    with st.expander("고급 설정 — 로브(lobe) 형태 가정치 (열복사·LSIR에 적용)"):
        back_frac = st.slider("풍상측 비율 (BACK_FRAC)", 0.0, 1.0, 0.30, 0.05)
        cross_frac = st.slider("좌우 폭 비율 (CROSS_FRAC)", 0.0, 1.0, 0.35, 0.05)

    # LSIR 계산 해상도는 1.00m 고정 (그 이상 세분화해도 의미 있는 차이가 없어 설정 제거)
    grid_res_m = 1.0

    st.markdown("---")
    st.caption(
        "필요 시트\n"
        "- 폭발: IS_Coordinates, PHAST_Distances, Wind rose(Meteorology)\n"
        "- 열복사: 폭발과 동일\n"
        "- LSIR: 위 + Leak_Frequencies, Vulnerability_Criteria, Ignition_Probability"
    )

if not image_file or not excel_file:
    st.info("왼쪽 사이드바에서 위성사진과 엑셀 파일을 모두 업로드해주세요.")
    st.stop()

image_bytes = image_file.getvalue()
excel_bytes = excel_file.getvalue()


def _detect_header_row(xls, sheet_name, max_scan_rows=6):
    """병합된 제목행 등으로 실제 컬럼명이 1행이 아닌 다른 행에 있는 경우를 대비해,
    상위 몇 개 행 중 '문자열 셀이 가장 많은 행'을 헤더 행으로 추정한다.
    일반적인 헤더 행은 거의 모든 칸이 텍스트 라벨이고, 병합된 제목행은 셀 1개만 채워져 있거나
    데이터행은 숫자가 섞여 있어 텍스트 칸 수가 헤더 행보다 적다."""
    raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    best_row, best_score = 0, -1
    for i in range(len(raw)):
        row = raw.iloc[i]
        str_count = sum(isinstance(v, str) and v.strip() != "" for v in row)
        if str_count > best_score:
            best_score, best_row = str_count, i
    return best_row


@st.cache_data(show_spinner=False)
def load_excel(_bytes):
    xls = pd.ExcelFile(io.BytesIO(_bytes))
    result = {}
    for name in xls.sheet_names:
        header_row = _detect_header_row(xls, name)
        df = pd.read_excel(xls, sheet_name=name, header=header_row)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
        result[name] = df
    return result


@st.cache_data(show_spinner=False)
def load_image(_bytes):
    return Image.open(io.BytesIO(_bytes)).convert("RGB")


try:
    sheets = load_excel(excel_bytes)
except Exception as e:
    st.error(f"엑셀 파일을 열 수 없습니다: {e}")
    st.stop()

try:
    base_img = load_image(image_bytes)
except Exception as e:
    st.error(f"이미지 파일을 열 수 없습니다: {e}")
    st.stop()


def missing_sheets(required):
    return [s for s in required if s not in sheets]


def get_xy_cols(df):
    x = next((c for c in df.columns if str(c).startswith("Loc_X")), None)
    y = next((c for c in df.columns if str(c).startswith("Loc_Y")), None)
    return x, y


def load_font():
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(p, 11)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_SMALL = load_font()


def m_to_px_factory(img_size, grid_extent_m):
    W, H = img_size
    px_per_m = W / grid_extent_m

    def m_to_px(x_m, y_m):
        return x_m * px_per_m, H - y_m * px_per_m

    return m_to_px, px_per_m


def draw_grid_lines(draw, img_size, cell_size_m, px_per_m, color=(255, 255, 255, 110), width=1):
    """이미지 위에 cell_size_m 간격의 격자선을 그린다."""
    W, H = img_size
    cell_px = cell_size_m * px_per_m
    if cell_px < 2:
        return  # 칸이 너무 촘촘하면(픽셀 단위 이하) 그리지 않음
    x = 0.0
    while x <= W:
        draw.line([(x, 0), (x, H)], fill=color, width=width)
        x += cell_px
    y = 0.0
    while y <= H:
        draw.line([(0, y), (W, y)], fill=color, width=width)
        y += cell_px


def to_float_or_none(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().upper()
    if s in ("NR", "NA", ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ------------------------------------------------------------------
# TAB 1: Explosion (Overpressure)
# ------------------------------------------------------------------
def render_explosion():
    required = ["IS_Coordinates", "PHAST_Distances", "Wind rose(Meteorology)"]
    miss = missing_sheets(required)
    if miss:
        st.warning(f"이 분석에 필요한 시트가 없습니다: {miss}")
        return

    is_coords = sheets["IS_Coordinates"].copy()
    phast = sheets["PHAST_Distances"].copy()
    wind_rose = sheets["Wind rose(Meteorology)"].copy()

    x_col, y_col = get_xy_cols(is_coords)
    errors = []
    if x_col is None or y_col is None:
        errors.append("IS_Coordinates 시트에 Loc_X_*/Loc_Y_* 컬럼이 없습니다.")
    for c in ["IS_ID", "Weather_Class", "Exp_0.3", "Exp_0.5"]:
        df_ = is_coords if c == "IS_ID" else phast
        if c not in df_.columns:
            errors.append(f"필수 컬럼 '{c}'가 없습니다.")
    if errors:
        st.error("\n".join(f"- {e}" for e in errors))
        return

    is_coords[x_col] = pd.to_numeric(is_coords[x_col], errors="coerce")
    is_coords[y_col] = pd.to_numeric(is_coords[y_col], errors="coerce")
    phast["Exp_0.3"] = pd.to_numeric(phast["Exp_0.3"], errors="coerce")
    phast["Exp_0.5"] = pd.to_numeric(phast["Exp_0.5"], errors="coerce")

    weather_prob = wind_rose.groupby("Weather_Class")["Probability"].sum().to_dict()
    by_weather = phast.groupby(["IS_ID", "Weather_Class"])[["Exp_0.3", "Exp_0.5"]].max().reset_index()

    rows = []
    for is_id, grp in by_weather.groupby("IS_ID"):
        grp_valid = grp.dropna(subset=["Exp_0.3", "Exp_0.5"], how="all")
        if grp_valid.empty:
            continue
        idx = grp_valid["Exp_0.5"].idxmax() if grp_valid["Exp_0.5"].notna().any() else grp_valid["Exp_0.3"].idxmax()
        governing = grp_valid.loc[idx]
        rows.append({
            "IS_ID": is_id,
            "Exp_0.3": grp_valid["Exp_0.3"].max(),
            "Exp_0.5": grp_valid["Exp_0.5"].max(),
            "Governing_Weather_Class": governing["Weather_Class"],
            "Governing_Probability": weather_prob.get(governing["Weather_Class"], np.nan),
        })
    envelope = pd.DataFrame(rows)
    if envelope.empty:
        st.error("PHAST_Distances에서 유효한 Exp_0.3 / Exp_0.5 값을 찾지 못했습니다.")
        return

    st.dataframe(envelope, use_container_width=True)

    img = base_img.copy()
    m_to_px, px_per_m = m_to_px_factory(img.size, grid_extent_m)
    points = {
        row["IS_ID"]: m_to_px(row[x_col], row[y_col])
        for _, row in is_coords.iterrows()
        if pd.notna(row[x_col]) and pd.notna(row[y_col])
    }

    def union_outline_contours(level_col):
        mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        for _, row in envelope.iterrows():
            is_id = row["IS_ID"]
            r_m = row[level_col]
            if pd.isna(r_m) or is_id not in points:
                continue
            cx, cy = points[is_id]
            r_px = int(round(r_m * px_per_m))
            if r_px <= 0:
                continue
            cv2.circle(mask, (int(round(cx)), int(round(cy))), r_px, 255, thickness=-1, lineType=cv2.LINE_AA)
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_outline_contours(draw, contours, color, width):
        for cnt in contours:
            if len(cnt) < 2:
                continue
            pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
            pts.append(pts[0])
            draw.line(pts, fill=color, width=width, joint="curve")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if show_grid_lines:
        draw_grid_lines(draw, img.size, cell_size_m, px_per_m)
    COLOR_03 = (255, 165, 0, 255)
    COLOR_05 = (220, 20, 20, 255)
    draw_outline_contours(draw, union_outline_contours("Exp_0.3"), COLOR_03, 3)
    draw_outline_contours(draw, union_outline_contours("Exp_0.5"), COLOR_05, 3)

    for _, row in envelope.iterrows():
        is_id = row["IS_ID"]
        if is_id not in points:
            continue
        cx, cy = points[is_id]
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=1)
        prob = row["Governing_Probability"]
        prob_txt = f"{prob*100:.0f}%" if pd.notna(prob) else "N/A"
        label = f"{is_id}\n[{row['Governing_Weather_Class']}, P={prob_txt}]"
        draw.multiline_text((cx + 6, cy - 26), label, fill=(255, 255, 255, 255), font=FONT_SMALL,
                             stroke_width=2, stroke_fill=(0, 0, 0, 255))

    combined = Image.alpha_composite(img.convert("RGBA"), overlay)
    legend_draw = ImageDraw.Draw(combined)
    lx, ly = 12, 12
    legend_items = [("Explosion 0.5 bar outline (severe)", COLOR_05),
                     ("Explosion 0.3 bar outline (moderate)", COLOR_03)]
    box_w, box_h = 420, 18 * len(legend_items) + 14 + 49
    legend_draw.rectangle([lx, ly, lx + box_w, ly + box_h], fill=(0, 0, 0, 165))
    for i, (label, color) in enumerate(legend_items):
        yy = ly + 8 + i * 18
        legend_draw.line([(lx + 8, yy + 6), (lx + 22, yy + 6)], fill=color, width=3)
        legend_draw.text((lx + 28, yy - 2), label, fill=(255, 255, 255, 255), font=FONT_SMALL)
    note_y = ly + 8 + len(legend_items) * 18 + 6
    legend_draw.text(
        (lx + 8, note_y),
        f"Grid: {cell_size_m:.1f}m/cell ({grid_extent_m:.0f}m x {grid_extent_m:.0f}m, "
        f"{round(grid_cell_count)}x{round(grid_cell_count)} cells).",
        fill=(230, 230, 230, 255), font=FONT_SMALL,
    )
    legend_draw.text((lx + 8, note_y + 15), "Overlapping circles merged into one outline.",
                      fill=(230, 230, 230, 255), font=FONT_SMALL)

    final_img = combined.convert("RGB")
    st.image(final_img, use_container_width=True)
    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("결과 이미지 다운로드 (PNG)", buf, "explosion_contour_map.png", "image/png", key="dl_explosion")


# ------------------------------------------------------------------
# TAB 2: Thermal Radiation (directional lobes)
# ------------------------------------------------------------------
def lobe_polygon_px(center_m, R_m, bearing_from_deg, back_frac, cross_frac, m_to_px, n=48):
    downwind_deg = (bearing_from_deg + 180.0) % 360.0
    rad = math.radians(downwind_deg)
    dx_down, dy_down = math.sin(rad), math.cos(rad)
    dx_cross, dy_cross = math.cos(rad), -math.sin(rad)
    semi_major = R_m * (1 + back_frac) / 2.0
    semi_minor = R_m * cross_frac
    offset = R_m * (1 - back_frac) / 2.0
    ellipse_cx = center_m[0] + dx_down * offset
    ellipse_cy = center_m[1] + dy_down * offset
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    local_x = semi_major * np.cos(theta)
    local_y = semi_minor * np.sin(theta)
    east = ellipse_cx + local_x * dx_down + local_y * dx_cross
    north = ellipse_cy + local_x * dy_down + local_y * dy_cross
    return [m_to_px(e, n_) for e, n_ in zip(east, north)]


def render_thermal():
    required = ["IS_Coordinates", "PHAST_Distances", "Wind rose(Meteorology)"]
    miss = missing_sheets(required)
    if miss:
        st.warning(f"이 분석에 필요한 시트가 없습니다: {miss}")
        return

    is_coords = sheets["IS_Coordinates"].copy()
    phast = sheets["PHAST_Distances"].copy()
    wind_rose = sheets["Wind rose(Meteorology)"].copy()

    x_col, y_col = get_xy_cols(is_coords)
    errors = []
    if x_col is None or y_col is None:
        errors.append("IS_Coordinates 시트에 Loc_X_*/Loc_Y_* 컬럼이 없습니다.")
    for c in ["IS_ID", "Weather_Class"]:
        if c not in phast.columns:
            errors.append(f"PHAST_Distances 시트에 '{c}' 컬럼이 없습니다.")
    for c in ["Weather_Class", "Angle_degree"]:
        if c not in wind_rose.columns:
            errors.append(f"Wind rose(Meteorology) 시트에 '{c}' 컬럼이 없습니다.")
    thermal_cols = [c for c in ["Jet_12.5", "Jet_37.5", "Pool_12.5", "Pool_37.5"] if c in phast.columns]
    if not thermal_cols:
        errors.append("PHAST_Distances에 Jet_12.5/Jet_37.5/Pool_12.5/Pool_37.5 컬럼이 없습니다.")
    if errors:
        st.error("\n".join(f"- {e}" for e in errors))
        return

    is_coords[x_col] = pd.to_numeric(is_coords[x_col], errors="coerce")
    is_coords[y_col] = pd.to_numeric(is_coords[y_col], errors="coerce")
    for c in ["Jet_12.5", "Jet_37.5", "Pool_12.5", "Pool_37.5"]:
        if c not in phast.columns:
            phast[c] = np.nan
        else:
            phast[c] = pd.to_numeric(phast[c], errors="coerce")

    by_weather = phast.groupby(["IS_ID", "Weather_Class"])[
        ["Jet_12.5", "Jet_37.5", "Pool_12.5", "Pool_37.5"]].max().reset_index()

    def fire_type_for(is_id):
        sub = by_weather[by_weather["IS_ID"] == is_id]
        has_jet = sub[["Jet_12.5", "Jet_37.5"]].notna().any().any()
        has_pool = sub[["Pool_12.5", "Pool_37.5"]].notna().any().any()
        return "Jet fire" if has_jet else ("Pool fire" if has_pool else None)

    fire_types = {is_id: fire_type_for(is_id) for is_id in is_coords["IS_ID"]}

    img = base_img.copy()
    m_to_px, px_per_m = m_to_px_factory(img.size, grid_extent_m)
    points_m = {
        row["IS_ID"]: (row[x_col], row[y_col])
        for _, row in is_coords.iterrows()
        if pd.notna(row[x_col]) and pd.notna(row[y_col])
    }

    def build_union_mask(level_col):
        mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        for _, wr_row in wind_rose.iterrows():
            wc = wr_row["Weather_Class"]
            bearing = wr_row["Angle_degree"]
            if pd.isna(bearing):
                continue
            matches = by_weather[by_weather["Weather_Class"] == wc]
            for _, row in matches.iterrows():
                is_id = row["IS_ID"]
                ftype = fire_types.get(is_id)
                if ftype is None or is_id not in points_m:
                    continue
                col = level_col.replace("LEVEL", "Jet" if ftype == "Jet fire" else "Pool")
                R = row[col]
                if pd.isna(R) or R <= 0:
                    continue
                poly = lobe_polygon_px(points_m[is_id], R, bearing, back_frac, cross_frac, m_to_px)
                poly_int = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
                cv2.fillPoly(mask, [poly_int], 255, lineType=cv2.LINE_AA)
        return mask

    def mask_to_contours(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_outline_contours(draw, contours, color, width):
        for cnt in contours:
            if len(cnt) < 2:
                continue
            pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
            pts.append(pts[0])
            draw.line(pts, fill=color, width=width, joint="curve")

    mask_125 = build_union_mask("LEVEL_12.5")
    mask_375 = build_union_mask("LEVEL_37.5")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if show_grid_lines:
        draw_grid_lines(draw, img.size, cell_size_m, px_per_m)
    COLOR_125 = (255, 200, 0, 255)
    COLOR_375 = (220, 20, 20, 255)
    draw_outline_contours(draw, mask_to_contours(mask_125), COLOR_125, 3)
    draw_outline_contours(draw, mask_to_contours(mask_375), COLOR_375, 3)

    for is_id, (x_m, y_m) in points_m.items():
        ftype = fire_types.get(is_id)
        if ftype is None:
            continue
        cx, cy = m_to_px(x_m, y_m)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=1)
        draw.text((cx + 6, cy - 14), f"{is_id} [{ftype}]", fill=(255, 255, 255, 255), font=FONT_SMALL,
                   stroke_width=2, stroke_fill=(0, 0, 0, 255))

    combined = Image.alpha_composite(img.convert("RGBA"), overlay)
    legend_draw = ImageDraw.Draw(combined)
    lx, ly = 12, 12
    legend_items = [("37.5 kW/m\u00b2 outline (severe)", COLOR_375),
                     ("12.5 kW/m\u00b2 outline (moderate)", COLOR_125)]
    box_w, box_h = 460, 18 * len(legend_items) + 14 + 48
    legend_draw.rectangle([lx, ly, lx + box_w, ly + box_h], fill=(0, 0, 0, 175))
    for i, (label, color) in enumerate(legend_items):
        yy = ly + 8 + i * 18
        legend_draw.line([(lx + 8, yy + 6), (lx + 22, yy + 6)], fill=color, width=3)
        legend_draw.text((lx + 28, yy - 2), label, fill=(255, 255, 255, 255), font=FONT_SMALL)
    note_y = ly + 8 + len(legend_items) * 18 + 6
    legend_draw.text((lx + 8, note_y), "Directional lobes per Wind rose (ASSUMED shape, not raw PHAST output):",
                      fill=(230, 230, 230, 255), font=FONT_SMALL)
    legend_draw.text((lx + 8, note_y + 15),
                      f"  downwind=R(PHAST), upwind={back_frac:.0%} R, crosswind half-width={cross_frac:.0%} R",
                      fill=(230, 230, 230, 255), font=FONT_SMALL)
    legend_draw.text(
        (lx + 8, note_y + 30),
        f"Grid: {cell_size_m:.1f}m/cell ({grid_extent_m:.0f}m x {grid_extent_m:.0f}m, "
        f"{round(grid_cell_count)}x{round(grid_cell_count)} cells).",
        fill=(230, 230, 230, 255), font=FONT_SMALL,
    )

    final_img = combined.convert("RGB")
    st.image(final_img, use_container_width=True)
    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("결과 이미지 다운로드 (PNG)", buf, "thermal_contour_directional.png", "image/png", key="dl_thermal")


# ------------------------------------------------------------------
# TAB 3: LSIR
# ------------------------------------------------------------------
def circle_mask(X, Y, center, r):
    if r is None:
        return None
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return dist <= r


def lobe_mask(X, Y, center, r, bearing_from_deg, back_frac, cross_frac):
    if r is None or r <= 0:
        return None
    downwind_deg = (bearing_from_deg + 180.0) % 360.0
    rad = math.radians(downwind_deg)
    dx_down, dy_down = math.sin(rad), math.cos(rad)
    dx_cross, dy_cross = math.cos(rad), -math.sin(rad)
    semi_major = r * (1 + back_frac) / 2.0
    semi_minor = r * cross_frac
    offset = r * (1 - back_frac) / 2.0
    ellipse_cx = center[0] + dx_down * offset
    ellipse_cy = center[1] + dy_down * offset
    rel_x = X - ellipse_cx
    rel_y = Y - ellipse_cy
    local_along = rel_x * dx_down + rel_y * dy_down
    local_cross = rel_x * dx_cross + rel_y * dy_cross
    return (local_along / semi_major) ** 2 + (local_cross / semi_minor) ** 2 <= 1.0


def add_band_contribution(risk_field, mask_severe, mask_mild, v_severe, v_mild, weight):
    if weight <= 0:
        return
    base = 0.0
    if mask_mild is not None:
        risk_field[mask_mild] += weight * v_mild
        base = v_mild
    if mask_severe is not None:
        risk_field[mask_severe] += weight * (v_severe - base)


def add_single_contribution(risk_field, mask, v, weight):
    if weight <= 0 or mask is None:
        return
    risk_field[mask] += weight * v


def find_col_by_keywords(df, keywords, exclude=()):
    """컬럼 이름에 keywords 중 하나가 포함된 컬럼을 찾는다 (공백/언더바 무시, 대소문자 무시)."""
    for c in df.columns:
        if c in exclude:
            continue
        norm = str(c).strip().lower().replace(" ", "").replace("_", "")
        if any(k in norm for k in keywords):
            return c
    return None


def find_hole_size_col(df, exclude_cols):
    """PHAST_Distances / Ignition_Probability 시트에서 hole size(누출구경) 컬럼을 찾는다.
    1) 이름에 hole/leak + size/diam 이 들어간 컬럼을 우선 탐색 (템플릿마다 이름이 다를 수 있음)
    2) 못 찾으면 IS_ID/Weather_Class/기타 지정 컬럼을 뺀 나머지 중 '대부분 숫자'인
       첫 번째 컬럼을 hole size 컬럼으로 추정 (원본 스크립트가 위치 기반으로 읽던 방식과 동일한 발상)."""
    keywords = ["holesize", "leaksize", "holediam", "leakdiam", "diameter", "hole_size", "leak_size"]
    for c in df.columns:
        norm = str(c).strip().lower().replace(" ", "").replace("_", "")
        if any(k.replace("_", "") in norm for k in keywords):
            return c
    for c in df.columns:
        if c in exclude_cols:
            continue
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().mean() > 0.8:
            return c
    return None


def parse_number_from_text(c):
    """'Freq_2mm', '5.4', 'Hole_22.3' 처럼 컬럼명에 숫자가 문자와 섞여 있어도
    그 안의 첫 숫자를 뽑아낸다 (컬럼명 전체를 float()로 바로 변환할 수 없는 경우 대비)."""
    s = str(c)
    try:
        return float(s)
    except ValueError:
        pass
    m = re.search(r"(\d+\.?\d*)", s)
    return float(m.group(1)) if m else None


def find_id_col(df):
    """IS_ID 컬럼을 찾는다. 이름으로 못 찾으면 첫 번째 컬럼을 ID로 간주한다
    (원본 스크립트가 항상 '첫 컬럼 = IS_ID' 위치로 읽었던 것과 동일한 가정)."""
    exact = find_col_by_keywords(df, ["is_id", "isid"])
    if exact is not None:
        return exact
    return df.columns[0] if len(df.columns) > 0 else None


@st.cache_data(show_spinner=False)
def compute_lsir_field(_excel_bytes, grid_extent_m, grid_res_m, back_frac, cross_frac):
    sheets_ = load_excel(_excel_bytes)
    is_coords_df = sheets_["IS_Coordinates"].copy()
    leak_freq_df = sheets_["Leak_Frequencies"].copy()
    phast_df = sheets_["PHAST_Distances"].copy()
    wind_rose_df = sheets_["Wind rose(Meteorology)"].copy()
    vuln_df = sheets_["Vulnerability_Criteria"].copy()
    ignition_df = sheets_["Ignition_Probability"].copy()

    x_col = next((c for c in is_coords_df.columns if str(c).startswith("Loc_X")), None)
    y_col = next((c for c in is_coords_df.columns if str(c).startswith("Loc_Y")), None)
    id_col_coords = find_id_col(is_coords_df)
    is_coords_df[x_col] = pd.to_numeric(is_coords_df[x_col], errors="coerce")
    is_coords_df[y_col] = pd.to_numeric(is_coords_df[y_col], errors="coerce")
    is_coords = {
        row[id_col_coords]: (row[x_col], row[y_col])
        for _, row in is_coords_df.iterrows()
        if pd.notna(row[x_col]) and pd.notna(row[y_col])
    }

    id_col_leak = find_id_col(leak_freq_df)
    hole_size_cols = [c for c in leak_freq_df.columns if c != id_col_leak]
    hole_sizes = []
    for c in hole_size_cols:
        hs = parse_number_from_text(c)
        if hs is not None:
            hole_sizes.append((c, hs))

    leak_freq = {}
    for _, row in leak_freq_df.iterrows():
        is_id = row[id_col_leak]
        leak_freq[is_id] = {hs: (float(row[col]) if pd.notna(row[col]) else 0.0) for col, hs in hole_sizes}

    thermal_level_cols = [c for c in ["Jet_12.5", "Jet_37.5", "Pool_12.5", "Pool_37.5",
                                        "Flash_LFL", "Exp_0.3", "Exp_0.5"] if c in phast_df.columns]
    id_col_phast = find_id_col(phast_df)
    hole_col_phast = find_hole_size_col(
        phast_df, exclude_cols={id_col_phast, "Weather_Class"} | set(thermal_level_cols)
    )
    if hole_col_phast is None:
        raise KeyError(
            "PHAST_Distances 시트에서 hole size(누출구경) 컬럼을 찾지 못했습니다. "
            "IS_ID/Weather_Class 외에 숫자로 된 hole size 컬럼이 있는지 확인해주세요."
        )
    phast = {}
    for _, row in phast_df.iterrows():
        hs_val = to_float_or_none(row[hole_col_phast])
        if hs_val is None:
            continue
        key = (row[id_col_phast], hs_val, row["Weather_Class"])
        phast[key] = {c: to_float_or_none(row[c]) for c in thermal_level_cols}
        for c in ["Jet_12.5", "Jet_37.5", "Pool_12.5", "Pool_37.5", "Flash_LFL", "Exp_0.3", "Exp_0.5"]:
            phast[key].setdefault(c, None)

    by_is = phast_df.groupby(id_col_phast)

    def classify_is(is_id):
        if is_id not in by_is.groups:
            return {"has_jet": False, "has_pool": False}
        sub = by_is.get_group(is_id)
        has_jet = any(sub[c].apply(lambda v: to_float_or_none(v) is not None).any()
                      for c in ["Jet_12.5", "Jet_37.5"] if c in sub.columns)
        has_pool = any(sub[c].apply(lambda v: to_float_or_none(v) is not None).any()
                       for c in ["Pool_12.5", "Pool_37.5"] if c in sub.columns)
        return {"has_jet": has_jet, "has_pool": has_pool}

    is_classification = {is_id: classify_is(is_id) for is_id in is_coords}

    wind_rose_df["Angle_degree"] = pd.to_numeric(wind_rose_df["Angle_degree"], errors="coerce")
    wind_rose_df["Probability"] = pd.to_numeric(wind_rose_df["Probability"], errors="coerce")
    wind_rose_df = wind_rose_df.dropna(subset=["Probability", "Angle_degree"])
    total_prob = wind_rose_df["Probability"].sum()
    wind_rows = [
        {"weather_class": row["Weather_Class"], "angle_deg": row["Angle_degree"],
         "prob": row["Probability"] / total_prob}
        for _, row in wind_rose_df.iterrows()
    ] if total_prob > 0 else []

    hazard_col = find_col_by_keywords(vuln_df, ["hazard"])
    intensity_col = find_col_by_keywords(vuln_df, ["intens", "level", "band"])
    outdoor_col = find_col_by_keywords(vuln_df, ["outdoor"])
    # 이름으로 못 찾으면 원본 템플릿의 컬럼 순서(hazard, intensity, unit, outdoor, indoor) 가정
    cols = list(vuln_df.columns)
    if hazard_col is None and len(cols) > 0:
        hazard_col = cols[0]
    if intensity_col is None and len(cols) > 1:
        intensity_col = cols[1]
    if outdoor_col is None and len(cols) > 3:
        outdoor_col = cols[3]
    if hazard_col is None or intensity_col is None or outdoor_col is None:
        raise KeyError(
            "Vulnerability_Criteria 시트에서 Hazard_Type / Intensity / Outdoor_Vulnerability에 "
            "해당하는 컬럼을 찾지 못했습니다."
        )

    vuln = {}
    for _, row in vuln_df.iterrows():
        vuln[(str(row[hazard_col]).strip(), str(row[intensity_col]).strip())] = float(row[outdoor_col])

    def vuln_lookup(hazard_type, intensity, default=0.0):
        return vuln.get((hazard_type, intensity), default)

    v_thermal_375 = vuln_lookup("Thermal (Jet/Pool)", "37.5")
    v_thermal_125 = vuln_lookup("Thermal (Jet/Pool)", "12.5")
    v_flash = vuln_lookup("Flash Fire", "5 (Methane LFL)")
    v_exp_05 = vuln_lookup("Overpressure", "0.5")
    v_exp_03 = vuln_lookup("Overpressure", "0.3")
    if all(v == 0.0 for v in [v_thermal_375, v_thermal_125, v_flash, v_exp_05, v_exp_03]):
        found_pairs = sorted(set(vuln.keys()))
        raise KeyError(
            "Vulnerability_Criteria의 Hazard_Type/Intensity 값이 예상한 문구와 달라 하나도 매칭되지 않았습니다. "
            f"시트에서 실제로 발견된 (Hazard_Type, Intensity) 조합: {found_pairs}"
        )

    id_col_ignition = find_id_col(ignition_df)
    hole_col_ignition = find_hole_size_col(
        ignition_df, exclude_cols={id_col_ignition, "P_jet", "P_flash", "P_vce", "P_pool"}
    )
    if hole_col_ignition is None:
        raise KeyError(
            "Ignition_Probability 시트에서 hole size(누출구경) 컬럼을 찾지 못했습니다."
        )
    ignition = {}
    for _, row in ignition_df.iterrows():
        is_id, hs = row[id_col_ignition], row[hole_col_ignition]
        if pd.isna(is_id) or pd.isna(hs):
            continue
        ignition[(is_id, float(hs))] = {
            "P_jet": row.get("P_jet", 0.0) or 0.0,
            "P_flash": row.get("P_flash", 0.0) or 0.0,
            "P_vce": row.get("P_vce", 0.0) or 0.0,
            "P_pool": row.get("P_pool", 0.0) or 0.0,
        }

    n = int(grid_extent_m / grid_res_m) + 1
    xs = np.linspace(0, grid_extent_m, n)
    ys = np.linspace(0, grid_extent_m, n)
    X, Y = np.meshgrid(xs, ys)
    risk_field = np.zeros_like(X, dtype=float)

    for is_id, center in is_coords.items():
        cls = is_classification.get(is_id, {"has_jet": False, "has_pool": False})
        for _, hs in hole_sizes:
            freq = leak_freq.get(is_id, {}).get(hs, 0.0)
            if freq <= 0:
                continue
            ign = ignition.get((is_id, hs))
            if ign is None:
                continue
            for wr in wind_rows:
                wc, bearing, wind_p = wr["weather_class"], wr["angle_deg"], wr["prob"]
                d = phast.get((is_id, hs, wc))
                if d is None:
                    continue
                if cls["has_pool"]:
                    weight = freq * wind_p * ign["P_pool"]
                    ms = lobe_mask(X, Y, center, d["Pool_37.5"], bearing, back_frac, cross_frac)
                    mm = lobe_mask(X, Y, center, d["Pool_12.5"], bearing, back_frac, cross_frac)
                    add_band_contribution(risk_field, ms, mm, v_thermal_375, v_thermal_125, weight)
                if cls["has_jet"]:
                    weight_jet = freq * wind_p * ign["P_jet"]
                    ms = lobe_mask(X, Y, center, d["Jet_37.5"], bearing, back_frac, cross_frac)
                    mm = lobe_mask(X, Y, center, d["Jet_12.5"], bearing, back_frac, cross_frac)
                    add_band_contribution(risk_field, ms, mm, v_thermal_375, v_thermal_125, weight_jet)

                    weight_flash = freq * wind_p * ign["P_flash"]
                    mf = lobe_mask(X, Y, center, d["Flash_LFL"], bearing, back_frac, cross_frac)
                    add_single_contribution(risk_field, mf, v_flash, weight_flash)

                    weight_vce = freq * wind_p * ign["P_vce"]
                    mse = circle_mask(X, Y, center, d["Exp_0.5"])
                    mme = circle_mask(X, Y, center, d["Exp_0.3"])
                    add_band_contribution(risk_field, mse, mme, v_exp_05, v_exp_03, weight_vce)

    return X, Y, risk_field, is_coords


def render_lsir():
    required = ["IS_Coordinates", "Leak_Frequencies", "PHAST_Distances",
                "Wind rose(Meteorology)", "Vulnerability_Criteria", "Ignition_Probability"]
    miss = missing_sheets(required)
    if miss:
        st.warning(f"이 분석에 필요한 시트가 없습니다: {miss}")
        return

    n_est = int(grid_extent_m / grid_res_m) + 1
    if n_est > 700:
        st.warning(f"해상도가 높습니다 ({n_est}x{n_est} 포인트) — 계산이 느릴 수 있습니다.")

    with st.spinner("LSIR 계산 중..."):
        try:
            X, Y, risk_field, is_coords = compute_lsir_field(
                excel_bytes, grid_extent_m, grid_res_m, back_frac, cross_frac
            )
        except Exception as e:
            st.error(f"LSIR 계산 중 오류가 발생했습니다 (엑셀 컬럼명이 템플릿과 다를 수 있습니다): {e}")
            return

    st.write(f"Max LSIR: {risk_field.max():.3e} /yr  |  Min LSIR: {risk_field.min():.3e} /yr")

    plot_field = np.clip(risk_field, 1e-12, None)
    levels = [1e-6, 1e-5, 1e-4, 1e-3]
    colors = ["#2ca02c", "#ffff33", "#ff9900", "#ff0000"]

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    ax.imshow(np.asarray(base_img), extent=[0, grid_extent_m, 0, grid_extent_m], origin="upper")
    if show_grid_lines:
        ticks = np.arange(0, grid_extent_m + 1e-6, cell_size_m)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, color="white", alpha=0.35, linewidth=0.5)
        ax.tick_params(labelsize=6)

    if plot_field.max() > 1e-12:
        try:
            ax.contour(X, Y, plot_field, levels=levels, norm=LogNorm(), colors=colors, linewidths=1.8)
            legend_handles = [Line2D([0], [0], color=c, lw=2) for c in colors]
            ax.legend(legend_handles, [f"{lv:.0e} /yr" for lv in levels], loc="lower left",
                       fontsize=8, framealpha=0.85, title="LSIR level")
        except Exception as e:
            st.warning(f"등고선을 그리지 못했습니다: {e}")
    else:
        st.warning("계산된 위험도가 전 영역에서 0에 가깝습니다 — 입력 데이터를 확인해주세요.")

    for is_id, (x, y) in is_coords.items():
        ax.plot(x, y, "o", color="deeppink", markersize=5)
        ax.annotate(is_id, (x, y), fontsize=6, color="white", textcoords="offset points", xytext=(4, 4))

    ax.set_xlim(0, grid_extent_m)
    ax.set_ylim(0, grid_extent_m)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("LSIR Contour (1/year)")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button("결과 이미지 다운로드 (PNG)", buf, "lsir_contour.png", "image/png", key="dl_lsir")
    plt.close(fig)


# ------------------------------------------------------------------
# 탭 렌더링
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["💥 Explosion (Overpressure)", "🔥 Thermal Radiation", "☠️ LSIR"])

with tab1:
    render_explosion()

with tab2:
    render_thermal()

with tab3:
    render_lsir()
