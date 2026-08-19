# -*- coding: utf-8 -*-
"""
QRA Risk Contour Suite (통합 앱)
- 위성사진 + QRA 엑셀 파일을 사이드바에서 한 번만 업로드
- st.tabs로 3가지 결과(Explosion / Thermal / LSIR)를 한 화면에서 확인
- 각 탭은 필요한 시트가 있는지 개별적으로 검증 -> 없으면 그 탭만 경고, 나머지는 정상 작동
"""

import io
import json
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

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

# ------------------------------------------------------------------
# Streamlit 버전 호환 레이어: use_container_width 지원 여부를 실행 시점에 확인해서,
# 지원하지 않는 구버전에서는 자동으로 use_column_width(구버전 이름) 등으로 대체한다.
# (버전을 바꿔도 이 부분이 자동으로 맞춰주므로 TypeError가 나지 않는다.)
# ------------------------------------------------------------------
import inspect


def _widget_supports(func, kwarg):
    try:
        return kwarg in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _width_kwarg(func):
    if _widget_supports(func, "use_container_width"):
        return {"use_container_width": True}
    if _widget_supports(func, "use_column_width"):
        return {"use_column_width": True}
    return {}


IMAGE_WIDTH_KW = _width_kwarg(st.image)
PYPLOT_WIDTH_KW = _width_kwarg(st.pyplot)
DATAFRAME_WIDTH_KW = _width_kwarg(st.dataframe)
DATA_EDITOR_WIDTH_KW = _width_kwarg(st.data_editor)

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
        "- LSIR: 위 + Leak_Frequencies, Vulnerability_Criteria, Ignition_Probability\n"
        "- F-N Curve: LSIR과 동일 + Occupancy "
        "(Area는 F-N Curve 탭에서 위성사진 위에 직접 그립니다)"
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

    wind_rose["Probability"] = pd.to_numeric(wind_rose["Probability"], errors="coerce")
    total_prob_wr = wind_rose["Probability"].sum()
    weather_prob_raw = wind_rose.groupby("Weather_Class")["Probability"].sum().to_dict()
    weather_prob = (
        {wc: p / total_prob_wr for wc, p in weather_prob_raw.items()}
        if total_prob_wr and total_prob_wr > 0 else weather_prob_raw
    )
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

    st.dataframe(envelope, **DATAFRAME_WIDTH_KW)

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
    st.image(final_img, **IMAGE_WIDTH_KW)
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

    if "Probability" in wind_rose.columns:
        wind_rose["Probability"] = pd.to_numeric(wind_rose["Probability"], errors="coerce")
        total_prob_wr = wind_rose["Probability"].sum()
        weather_prob_raw = wind_rose.groupby("Weather_Class")["Probability"].sum().to_dict()
        weather_prob = (
            {wc: p / total_prob_wr for wc, p in weather_prob_raw.items()}
            if total_prob_wr and total_prob_wr > 0 else weather_prob_raw
        )
    else:
        weather_prob = {}

    thermal_rows = []
    for is_id, ftype in fire_types.items():
        if ftype is None:
            continue
        col_375 = "Jet_37.5" if ftype == "Jet fire" else "Pool_37.5"
        col_125 = "Jet_12.5" if ftype == "Jet fire" else "Pool_12.5"
        sub = by_weather[by_weather["IS_ID"] == is_id]
        sub_valid = sub.dropna(subset=[col_375, col_125], how="all")
        if sub_valid.empty:
            continue
        idx = (sub_valid[col_375].idxmax() if sub_valid[col_375].notna().any()
               else sub_valid[col_125].idxmax())
        governing = sub_valid.loc[idx]
        thermal_rows.append({
            "IS_ID": is_id,
            "Fire_Type": ftype,
            "R_37.5(m)": sub_valid[col_375].max(),
            "R_12.5(m)": sub_valid[col_125].max(),
            "Governing_Weather_Class": governing["Weather_Class"],
            "Governing_Probability": weather_prob.get(governing["Weather_Class"], np.nan),
        })
    thermal_envelope = pd.DataFrame(thermal_rows)
    if not thermal_envelope.empty:
        st.dataframe(thermal_envelope, **DATAFRAME_WIDTH_KW)
    else:
        st.info("표를 만들 유효한 Jet/Pool 열복사 거리 데이터가 없습니다.")

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
    st.image(final_img, **IMAGE_WIDTH_KW)
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

    st.pyplot(fig, **PYPLOT_WIDTH_KW)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button("결과 이미지 다운로드 (PNG)", buf, "lsir_contour.png", "image/png", key="dl_lsir")
    plt.close(fig)


# ------------------------------------------------------------------
# TAB 4: F-N Curve (이산확률/이항분포 방식, 정수 N)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# TAB 4: F-N Curve (이산확률/이항분포 방식, 정수 N)
# Area는 엑셀이 아니라 위성사진 위에서 사용자가 직접 사각형으로 그려서 정의한다.
# ------------------------------------------------------------------
def _norm_name(s):
    """Area 이름 비교용 정규화: 공백/언더바/대소문자 차이를 무시한다."""
    return re.sub(r"[\s_]+", "", str(s).strip().lower())


def binom_pmf_array(n, p):
    """이항분포 B(n, p)의 PMF를 정수 k=0..n에 대해 반환한다 (log-공간 계산으로 큰 n에서도 안정적)."""
    n = int(round(n))
    if n <= 0 or p is None or p <= 0:
        return np.array([1.0])
    p = min(max(float(p), 0.0), 1.0)
    if p >= 1.0:
        arr = np.zeros(n + 1)
        arr[n] = 1.0
        return arr
    ks = np.arange(n + 1)
    log_coeff = np.array([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in ks])
    log_pmf = log_coeff + ks * math.log(p) + (n - ks) * math.log(1.0 - p)
    pmf = np.exp(log_pmf)
    total = pmf.sum()
    if total > 0:
        pmf = pmf / total
    return pmf


def rotated_rect_sample_points(cx, cy, w, h, angle_deg, n_samples=12):
    """중심(cx,cy), 가로/세로(w,h), 회전각(angle_deg, 반시계=+, m 좌표계 기준)을 가진
    사각형 내부를 n_samples x n_samples 격자로 샘플링한 (X,Y) 좌표 배열을 반환한다."""
    hw, hh = w / 2.0, h / 2.0
    xs = np.linspace(-hw, hw, n_samples)
    ys = np.linspace(-hh, hh, n_samples)
    LX, LY = np.meshgrid(xs, ys)
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    WX = cx + LX * cos_t - LY * sin_t
    WY = cy + LX * sin_t + LY * cos_t
    return WX, WY


def rect_band_fractions(cx, cy, w, h, angle_deg, center, r_severe, r_mild, bearing,
                         mode, back_frac, cross_frac, n_samples=12):
    """(회전 가능한) 사각형 영역을 n_samples x n_samples 격자로 샘플링해, 심각(severe)/경미(mild)
    밴드에 걸치는 면적비를 근사 계산한다 (면적비 기반 정밀 판정)."""
    XX, YY = rotated_rect_sample_points(cx, cy, w, h, angle_deg, n_samples)
    if mode == "lobe":
        sev = lobe_mask(XX, YY, center, r_severe, bearing, back_frac, cross_frac)
        mld = lobe_mask(XX, YY, center, r_mild, bearing, back_frac, cross_frac)
    else:
        sev = circle_mask(XX, YY, center, r_severe)
        mld = circle_mask(XX, YY, center, r_mild)
    total = XX.size
    sev_arr = np.zeros_like(XX, dtype=bool) if sev is None else np.asarray(sev)
    mld_arr = np.zeros_like(XX, dtype=bool) if mld is None else np.asarray(mld)
    frac_severe = sev_arr.sum() / total
    frac_mild_only = (mld_arr & ~sev_arr).sum() / total
    return float(frac_severe), float(frac_mild_only)


def rect_single_fraction(cx, cy, w, h, angle_deg, center, r, bearing, mode,
                          back_frac, cross_frac, n_samples=12):
    """단일 밴드(예: Flash Fire LFL)에 대한 (회전 가능한) 사각형-면적 겹침비를 근사 계산한다."""
    if r is None or r <= 0:
        return 0.0
    XX, YY = rotated_rect_sample_points(cx, cy, w, h, angle_deg, n_samples)
    hit = lobe_mask(XX, YY, center, r, bearing, back_frac, cross_frac) if mode == "lobe" \
        else circle_mask(XX, YY, center, r)
    if hit is None:
        return 0.0
    hit = np.asarray(hit)
    return float(hit.sum() / hit.size)


def load_occupancy_wide(occ_df):
    """Personnel_Category | No_of_Personnel | Area1 | Area2 | ... 형태의 넓은 포맷
    Occupancy 시트를 (구역명, 인원수, 그 구역에 머무는 시간비율) 레코드 목록으로 변환한다."""
    cat_col = find_col_by_keywords(occ_df, ["personnelcategory", "category", "personnel"])
    if cat_col is None:
        cat_col = occ_df.columns[0]
    pop_col = find_col_by_keywords(occ_df, ["nopersonnel", "population", "headcount", "personnel"],
                                    exclude=(cat_col,))
    if pop_col is None:
        remaining = [c for c in occ_df.columns if c != cat_col]
        pop_col = remaining[0] if remaining else None
    if pop_col is None:
        raise KeyError("Occupancy 시트에서 인원수(No_of_Personnel) 컬럼을 찾지 못했습니다.")
    area_cols = [c for c in occ_df.columns if c not in (cat_col, pop_col)]

    records = []
    for _, row in occ_df.iterrows():
        cat = str(row[cat_col]).strip()
        pop = to_float_or_none(row[pop_col])
        if pop is None or pop <= 0:
            continue
        for ac in area_cols:
            frac = to_float_or_none(row[ac])
            if frac is None or frac <= 0:
                continue
            records.append({"category": cat, "population": pop,
                             "area_name": str(ac).strip(), "time_fraction": frac})
    return records


def _area_corners_px(cx, cy, w, h, angle_deg, px_per_m, canvas_size):
    """m 좌표계의 (중심, 가로세로, 회전각)을 가진 사각형의 4개 꼭짓점을
    캔버스 픽셀 좌표로 변환한다."""
    hw, hh = w / 2.0, h / 2.0
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    pts_px = []
    for lx, ly in local:
        wx = cx + lx * cos_t - ly * sin_t
        wy = cy + lx * sin_t + ly * cos_t
        px = wx * px_per_m
        py = canvas_size - wy * px_per_m
        pts_px.append((px, py))
    return pts_px


def draw_areas_overlay(bg_img, areas_records, canvas_size, grid_extent_m, pending_corner=None):
    """지금까지 등록된 Area 사각형(+이름)과, 클릭 대기 중인 첫 모서리 표시를
    위성사진 위에 그려서 보여준다."""
    img = bg_img.convert("RGBA").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    px_per_m = canvas_size / grid_extent_m
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for a in areas_records:
        try:
            cx, cy = float(a["Loc_X_center"]), float(a["Loc_Y_center"])
            w, h = float(a["Width_m"]), float(a["Height_m"])
            angle = float(a.get("Rotation_deg", 0.0) or 0.0)
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        pts = _area_corners_px(cx, cy, w, h, angle, px_per_m, canvas_size)
        draw.polygon(pts, outline=(255, 102, 0, 255), fill=(255, 165, 0, 80), width=3)
        name = str(a.get("Area_Name", "")).strip()
        if name:
            cx_px = sum(p[0] for p in pts) / 4.0
            cy_px = sum(p[1] for p in pts) / 4.0
            draw.text((cx_px, cy_px), name, fill=(255, 0, 0, 255), font=font,
                      anchor="mm", stroke_width=2, stroke_fill=(255, 255, 255, 255))

    if pending_corner is not None:
        cx_m, cy_m = pending_corner
        px = cx_m * px_per_m
        py = canvas_size - cy_m * px_per_m
        r = 7
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0, 255), width=3)

    return Image.alpha_composite(img, overlay).convert("RGB")


def _migrate_area_record(a):
    """이전 포맷(Loc_X_min/max, Loc_Y_min/max, 회전 없음)으로 저장된 CSV를
    새 포맷(중심좌표+가로세로+회전각)으로 변환한다. 이미 새 포맷이면 그대로 통과시킨다."""
    if "Loc_X_center" in a and "Width_m" in a:
        return a
    try:
        x_min, x_max = float(a["Loc_X_min"]), float(a["Loc_X_max"])
        y_min, y_max = float(a["Loc_Y_min"]), float(a["Loc_Y_max"])
        return {
            "Area_Name": a.get("Area_Name", ""),
            "Loc_X_center": round((x_min + x_max) / 2, 1),
            "Loc_Y_center": round((y_min + y_max) / 2, 1),
            "Width_m": round(x_max - x_min, 1),
            "Height_m": round(y_max - y_min, 1),
            "Rotation_deg": 0.0,
        }
    except Exception:
        return a


def detect_rectangles_from_annotated_image(
    image_bytes,
    grid_extent_m,
    min_area_px=150,
    yellow_hsv_lower=(18, 80, 80),
    yellow_hsv_upper=(35, 255, 255),
    flip_rotation_sign=False,
):
    """PPT 등으로 위성사진 위에 '빨간 테두리 + 노란 배경' 직사각형을 그려 넣은 주석 이미지에서
    직사각형들을 인식해 (Area_Name, Loc_X_center, Loc_Y_center, Width_m, Height_m, Rotation_deg)
    레코드 목록으로 반환한다.

    좌표계는 이 앱의 기존 규약(원점=이미지 좌하단, y는 위쪽 방향이 양수)과 동일하게 맞춘다
    (m_to_px_factory, _area_corners_px 등에서 쓰는 것과 같은 px_per_m = W / grid_extent_m 규약).
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_px, w_px = img_bgr.shape[:2]
    px_per_m = w_px / grid_extent_m

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(yellow_hsv_lower, dtype=np.uint8)
    upper = np.array(yellow_hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    records = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        (cx_px, cy_px), (rw_px, rh_px), angle = cv2.minAreaRect(cnt)
        # OpenCV의 minAreaRect 각도 규약(버전별로 상이)을 보정: 긴 변을 Width로 통일
        if rw_px < rh_px:
            rw_px, rh_px = rh_px, rw_px
            angle += 90.0
        # cv2.minAreaRect의 각도는 '픽셀 좌표계(원점 왼쪽 위, y가 아래로 증가)' 기준이다.
        # 이 앱의 Area 좌표계는 '원점 왼쪽 아래, y가 위로 증가'라서 y축이 반전되어 있고,
        # x축은 그대로 두고 y축만 반전(거울반사)하면 회전의 부호(방향) 자체가 뒤집힌다.
        # 위치는 (h_px - cy_px)로 이미 y를 뒤집어 보정했으므로, 각도도 부호를 뒤집어야
        # 실제 방향과 일치한다 (기본으로 항상 적용).
        angle = -angle
        if flip_rotation_sign:
            # 그래도 방향이 반대로 보이면(예: cv2 버전별 각도 규약 차이) 체크박스로 다시 뒤집기
            angle = -angle

        records.append({
            "Area_Name": "",  # 아래 표에서 채움
            "Loc_X_center": round(cx_px / px_per_m, 1),
            "Loc_Y_center": round((h_px - cy_px) / px_per_m, 1),
            "Width_m": round(rw_px / px_per_m, 1),
            "Height_m": round(rh_px / px_per_m, 1),
            "Rotation_deg": round(angle, 1),
        })

    # 대략 왼쪽 위부터 보기 좋게 정렬 (y 내림차순, x 오름차순)
    records.sort(key=lambda r: (-r["Loc_Y_center"], r["Loc_X_center"]))
    return records, mask


def render_area_drawer():
    """위성사진을 클릭해서 Area 사각형을 정의하고(두 번 클릭 = 대각선 두 모서리),
    이름/회전각을 표에서 붙이고, CSV로 저장/불러올 수 있는 UI."""
    st.subheader("🖊️ Area(구역) 클릭으로 지정하기")
    st.caption(
        "① 아래 위성사진에서 사각형의 한쪽 모서리를 클릭하세요. "
        "② 이어서 반대쪽 대각선 모서리를 클릭하면 사각형 하나가 자동으로 표(사진 위쪽)에 추가됩니다. "
        "③ 표에서 Area_Name을 Occupancy 시트의 구역(컬럼)명과 맞게 고치고, "
        "필요하면 Rotation_deg(회전각, 도 단위)를 직접 입력해 실제 건물 방향에 맞춰주세요 — "
        "수정한 내용은 바로 아래 위성사진에 즉시 반영됩니다 "
        "(공백/언더바, 대소문자 차이는 자동으로 무시합니다)."
    )

    if "drawn_areas" not in st.session_state:
        st.session_state.drawn_areas = []
    if "areas_source_dirty" not in st.session_state:
        # True일 때만 st.session_state.drawn_areas(list)로부터 DataFrame을 새로 만든다.
        # 클릭으로 그리기/CSV 불러오기/이미지 인식 추가처럼 표 밖에서 목록이 바뀌었을 때만
        # True로 세팅한다 — 매 rerun마다 새 DataFrame을 만들어 같은 key의 data_editor에
        # 다시 넣으면, 방금 입력한 셀 수정이 "외부 데이터 변경"으로 오인되어 한 번에 반영되지
        # 않고 같은 칸을 두 번 고쳐야 반영되는 문제가 생기기 때문에 이렇게 분리한다.
        st.session_state.areas_source_dirty = True
    if "area_pending_corner" not in st.session_state:
        st.session_state.area_pending_corner = None
    if "area_click_last" not in st.session_state:
        st.session_state.area_click_last = None

    col_up, col_dl = st.columns(2)
    with col_up:
        uploaded_csv = st.file_uploader(
            "저장된 Area CSV 불러오기 (이전에 이 앱에서 내려받은 파일)", type=["csv"], key="area_csv_uploader"
        )
        if uploaded_csv is not None:
            try:
                df_loaded = pd.read_csv(uploaded_csv)
                records = [_migrate_area_record(r) for r in df_loaded.to_dict("records")]
                needed = {"Area_Name", "Loc_X_center", "Loc_Y_center", "Width_m", "Height_m", "Rotation_deg"}
                if not all(needed.issubset(set(r.keys())) for r in records):
                    st.error(f"CSV에 필요한 컬럼이 없습니다: {sorted(needed)}")
                else:
                    st.session_state.drawn_areas = records
                    st.session_state.areas_source_dirty = True
                    st.success(f"{len(records)}개 Area를 불러왔습니다.")
            except Exception as e:
                st.error(f"CSV를 읽을 수 없습니다: {e}")

    st.markdown("###### 🟨 이미지 인식으로 Area 일괄 추가 (선택)")
    st.caption(
        "PowerPoint 등에서 위성사진 위에 '빨간 테두리 + 노란 배경' 직사각형을 여러 개 그려 넣은 뒤 "
        "이미지로 내보내 업로드하면, 그 사각형들을 자동으로 인식해 아래 표에 추가합니다. "
        "(explosion/thermal/LSIR용으로 사이드바에 올린 순수 위성사진과는 별개의 파일이며, "
        "같은 격자 범위·해상도의 사진 위에 도형만 얹어서 내보내는 것을 권장합니다.)"
    )
    rect_img_file = st.file_uploader(
        "주석(직사각형 표시) 이미지 업로드 (png/jpg)", type=["png", "jpg", "jpeg"], key="rect_detect_uploader"
    )
    if rect_img_file is not None:
        with st.expander("고급 설정 (노란색 인식 민감도 / 회전 방향)"):
            rect_min_area = st.number_input(
                "최소 인식 면적(px²)", min_value=10, value=150, step=10, key="rect_min_area"
            )
            rect_h_lo, rect_h_hi = st.slider("노란색 Hue 범위", 0, 179, (18, 35), key="rect_hue_range")
            rect_flip_sign = st.checkbox(
                "회전각 부호 반전 (아래 미리보기에서 방향이 실제와 반대로 보이면 체크)",
                value=False, key="rect_flip_sign",
            )

        try:
            detected, mask_preview = detect_rectangles_from_annotated_image(
                rect_img_file.getvalue(), grid_extent_m,
                min_area_px=rect_min_area,
                yellow_hsv_lower=(rect_h_lo, 80, 80),
                yellow_hsv_upper=(rect_h_hi, 255, 255),
                flip_rotation_sign=rect_flip_sign,
            )
        except Exception as e:
            st.error(f"이미지를 분석할 수 없습니다: {e}")
            detected, mask_preview = [], None

        if rect_img_file is not None and not detected and mask_preview is not None:
            st.warning("직사각형이 인식되지 않았습니다. 노란색 Hue 범위나 최소 면적 값을 조정해보세요.")
        elif detected:
            existing_n = len(st.session_state.drawn_areas)
            for i, r in enumerate(detected, start=1):
                r["Area_Name"] = f"Area_{existing_n + i}"
            df_detected = pd.DataFrame(detected)
            st.success(f"{len(detected)}개의 직사각형을 인식했습니다. Area_Name을 확인/수정한 뒤 추가하세요.")
            edited_detected = st.data_editor(
                df_detected, num_rows="dynamic", key="rect_detected_editor", **DATA_EDITOR_WIDTH_KW
            )
            with st.expander("인식 마스크 미리보기 (디버깅용)"):
                st.image(mask_preview, caption="노란색 인식 영역 (흰색=인식됨)", **IMAGE_WIDTH_KW)

            if st.button("위 결과를 Area 목록에 추가", key="rect_detect_apply_btn"):
                st.session_state.drawn_areas.extend(edited_detected.to_dict("records"))
                st.session_state.areas_source_dirty = True
                st.success(f"{len(edited_detected)}개 Area가 목록에 추가되었습니다.")
                st.rerun()

    st.markdown("---")

    # 표를 사진보다 먼저 반영해야, 표에서 수정한 내용이 같은 실행(rerun) 안에서
    # 바로 아래 위성사진 오버레이에도 지연 없이 반영된다.
    if st.session_state.drawn_areas:
        if st.session_state.areas_source_dirty or "drawn_areas_df" not in st.session_state:
            # 표 밖에서 목록이 바뀐 경우에만 DataFrame을 새로 만든다 (위 areas_source_dirty 설명 참고).
            st.session_state.drawn_areas_df = pd.DataFrame(st.session_state.drawn_areas)
            st.session_state.areas_source_dirty = False

        edited = st.data_editor(
            st.session_state.drawn_areas_df, num_rows="dynamic", key="area_table_editor", **DATA_EDITOR_WIDTH_KW
        )
        # 에디터가 반환한 DataFrame 객체를 그대로 다음 rerun의 base로 재사용한다
        # (list로 왕복 변환하지 않아야 dtype/identity가 안정적으로 유지된다).
        st.session_state.drawn_areas_df = edited
        st.session_state.drawn_areas = edited.to_dict("records")

        with col_dl:
            csv_bytes = edited.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Area 목록 CSV로 저장", csv_bytes, "drawn_areas.csv", "text/csv", key="dl_areas_csv"
            )
    else:
        st.info("아직 지정한 Area가 없습니다. 아래 위성사진을 두 번 클릭하거나 위에서 CSV를 불러와주세요.")

    if not HAS_CLICK:
        st.error(
            "클릭으로 Area를 지정하는 기능을 쓰려면 requirements.txt에 "
            "`streamlit-image-coordinates`를 추가해야 합니다. (지금은 CSV 불러오기만 가능합니다.)"
        )
    else:
        canvas_size = 700
        bg_img_resized = base_img.resize((canvas_size, canvas_size))
        px_per_m_canvas = canvas_size / grid_extent_m

        if st.session_state.area_pending_corner is not None:
            st.info(
                f"첫 번째 모서리: ({st.session_state.area_pending_corner[0]:.1f}, "
                f"{st.session_state.area_pending_corner[1]:.1f}) m — 이제 반대쪽 모서리를 클릭하세요."
            )
            if st.button("첫 번째 모서리 취소", key="cancel_pending_corner"):
                st.session_state.area_pending_corner = None
                st.rerun()

        display_img = draw_areas_overlay(
            bg_img_resized, st.session_state.drawn_areas, canvas_size, grid_extent_m,
            st.session_state.area_pending_corner
        )
        click_value = streamlit_image_coordinates(display_img, key="area_click_img")

        if click_value is not None and click_value != st.session_state.area_click_last:
            st.session_state.area_click_last = click_value
            px, py = click_value["x"], click_value["y"]
            cx_click_m = px / px_per_m_canvas
            cy_click_m = (canvas_size - py) / px_per_m_canvas

            if st.session_state.area_pending_corner is None:
                st.session_state.area_pending_corner = (cx_click_m, cy_click_m)
                st.rerun()
            else:
                x1, y1 = st.session_state.area_pending_corner
                x2, y2 = cx_click_m, cy_click_m
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w, h = abs(x2 - x1), abs(y2 - y1)
                st.session_state.area_pending_corner = None
                if w > 0 and h > 0:
                    idx = len(st.session_state.drawn_areas) + 1
                    st.session_state.drawn_areas.append({
                        "Area_Name": f"Area_{idx}",
                        "Loc_X_center": round(cx, 1), "Loc_Y_center": round(cy, 1),
                        "Width_m": round(w, 1), "Height_m": round(h, 1),
                        "Rotation_deg": 0.0,
                    })
                    st.session_state.areas_source_dirty = True
                    st.success(f"사각형 추가: 중심({cx:.1f}, {cy:.1f})m, 가로{w:.1f}m x 세로{h:.1f}m")
                st.rerun()

    areas_out = []
    for a in st.session_state.drawn_areas:
        try:
            name = str(a["Area_Name"]).strip()
            cx, cy = float(a["Loc_X_center"]), float(a["Loc_Y_center"])
            w, h = float(a["Width_m"]), float(a["Height_m"])
            angle = float(a.get("Rotation_deg", 0.0) or 0.0)
            if not name or w <= 0 or h <= 0:
                continue
            areas_out.append({"name": name, "cx": cx, "cy": cy, "w": w, "h": h, "angle": angle})
        except Exception:
            continue
    return areas_out


@st.cache_data(show_spinner=False)
def compute_fn_curve(_excel_bytes, back_frac, cross_frac, areas_json):
    areas = json.loads(areas_json)

    sheets_ = load_excel(_excel_bytes)
    is_coords_df = sheets_["IS_Coordinates"].copy()
    leak_freq_df = sheets_["Leak_Frequencies"].copy()
    phast_df = sheets_["PHAST_Distances"].copy()
    wind_rose_df = sheets_["Wind rose(Meteorology)"].copy()
    vuln_df = sheets_["Vulnerability_Criteria"].copy()
    ignition_df = sheets_["Ignition_Probability"].copy()
    occ_df = sheets_["Occupancy"].copy()

    # --- IS_Coordinates (누출/점화원 위치) ---
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

    # --- Leak_Frequencies (홀크기별 누출빈도) ---
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

    # --- PHAST_Distances (결과 반경) ---
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

    # --- Wind rose (풍향/기상 확률) ---
    wind_rose_df["Angle_degree"] = pd.to_numeric(wind_rose_df["Angle_degree"], errors="coerce")
    wind_rose_df["Probability"] = pd.to_numeric(wind_rose_df["Probability"], errors="coerce")
    wind_rose_df = wind_rose_df.dropna(subset=["Probability", "Angle_degree"])
    total_prob = wind_rose_df["Probability"].sum()
    wind_rows = [
        {"weather_class": row["Weather_Class"], "angle_deg": row["Angle_degree"],
         "prob": row["Probability"] / total_prob}
        for _, row in wind_rose_df.iterrows()
    ] if total_prob > 0 else []

    # --- Vulnerability_Criteria (실외/실내 치사확률) ---
    hazard_col = find_col_by_keywords(vuln_df, ["hazard"])
    intensity_col = find_col_by_keywords(vuln_df, ["intens", "level", "band"])
    outdoor_col = find_col_by_keywords(vuln_df, ["outdoor"])
    indoor_col = find_col_by_keywords(vuln_df, ["indoor"])
    cols = list(vuln_df.columns)
    if hazard_col is None and len(cols) > 0:
        hazard_col = cols[0]
    if intensity_col is None and len(cols) > 1:
        intensity_col = cols[1]
    if outdoor_col is None and len(cols) > 3:
        outdoor_col = cols[3]
    if indoor_col is None and len(cols) > 4:
        indoor_col = cols[4]
    if hazard_col is None or intensity_col is None or outdoor_col is None:
        raise KeyError(
            "Vulnerability_Criteria 시트에서 Hazard_Type / Intensity / Outdoor_Vulnerability에 "
            "해당하는 컬럼을 찾지 못했습니다."
        )
    vuln_outdoor = {}
    vuln_indoor_fixed = {}  # Overpressure처럼 건물유형별로 갈라지는 항목은 여기 들어오지 않음(텍스트라 float 변환 실패)
    for _, row in vuln_df.iterrows():
        key = (str(row[hazard_col]).strip(), str(row[intensity_col]).strip())
        vuln_outdoor[key] = float(row[outdoor_col])
        if indoor_col is not None:
            fv = to_float_or_none(row[indoor_col])
            if fv is not None:
                vuln_indoor_fixed[key] = fv

    def vuln_lookup(hazard_type, intensity, default=0.0):
        return vuln_outdoor.get((hazard_type, intensity), default)

    v_thermal_375 = vuln_lookup("Thermal (Jet/Pool)", "37.5")
    v_thermal_125 = vuln_lookup("Thermal (Jet/Pool)", "12.5")
    v_flash = vuln_lookup("Flash Fire", "5 (Methane LFL)")
    v_exp_05 = vuln_lookup("Overpressure", "0.5")
    v_exp_03 = vuln_lookup("Overpressure", "0.3")
    if all(v == 0.0 for v in [v_thermal_375, v_thermal_125, v_flash, v_exp_05, v_exp_03]):
        found_pairs = sorted(set(vuln_outdoor.keys()))
        raise KeyError(
            "Vulnerability_Criteria의 Hazard_Type/Intensity 값이 예상한 문구와 달라 하나도 매칭되지 않았습니다. "
            f"시트에서 실제로 발견된 (Hazard_Type, Intensity) 조합: {found_pairs}"
        )

    # --- Indoor_Vuln_ByBuildingType (건물유형별 실내 과압 치사확률) + Area_Coordinates의 Building_Type ---
    # 시트 이름 뒤에 공백이 붙어 저장된 템플릿도 있어 이름을 앞뒤 공백 무시하고 찾는다.
    indoor_bldg_sheet = next(
        (sheets_[k] for k in sheets_ if k.strip() == "Indoor_Vuln_ByBuildingType"), None
    )
    indoor_vuln_bybldg = {}
    if indoor_bldg_sheet is not None:
        bt_col = find_col_by_keywords(indoor_bldg_sheet, ["buildingtype"])
        c03_col = find_col_by_keywords(indoor_bldg_sheet, ["0.3", "03bar"]) or next(
            (c for c in indoor_bldg_sheet.columns if "0.3" in str(c)), None
        )
        c05_col = find_col_by_keywords(indoor_bldg_sheet, ["0.5", "05bar"]) or next(
            (c for c in indoor_bldg_sheet.columns if "0.5" in str(c)), None
        )
        if bt_col is not None and c03_col is not None and c05_col is not None:
            for _, row in indoor_bldg_sheet.iterrows():
                bt = row[bt_col]
                v03, v05 = to_float_or_none(row[c03_col]), to_float_or_none(row[c05_col])
                if pd.isna(bt) or v03 is None or v05 is None:
                    continue
                indoor_vuln_bybldg[str(bt).strip()] = {"0.3": v03, "0.5": v05}

    area_coords_sheet = next(
        (sheets_[k] for k in sheets_ if k.strip() == "Area_Coordinates"), None
    )
    area_building_type = {}
    if area_coords_sheet is not None:
        name_col = find_col_by_keywords(area_coords_sheet, ["areaname"]) or area_coords_sheet.columns[0]
        bt_col2 = find_col_by_keywords(area_coords_sheet, ["buildingtype"])
        if bt_col2 is not None:
            for _, row in area_coords_sheet.iterrows():
                area_building_type[_norm_name(row[name_col])] = str(row[bt_col2]).strip()

    def get_vuln_for_area(hazard_type, intensity, area_norm_name, default=0.0):
        """구역의 Building_Type을 감안한 치사확률. 옥외/정보없음이면 Outdoor 값,
        실내이면 Overpressure는 건물유형별 표를, Thermal/Flash는 Vulnerability_Criteria의
        고정 Indoor_Vulnerability 값을 사용한다."""
        bt = area_building_type.get(area_norm_name)
        outdoor_v = vuln_outdoor.get((hazard_type, intensity), default)
        if not bt or "옥외" in bt:
            return outdoor_v
        if hazard_type == "Overpressure":
            b = indoor_vuln_bybldg.get(bt)
            if b is None:
                return outdoor_v
            band = "0.3" if "0.3" in intensity else "0.5"
            return b.get(band, outdoor_v)
        return vuln_indoor_fixed.get((hazard_type, intensity), outdoor_v)

    # --- Ignition_Probability (착화확률) ---
    id_col_ignition = find_id_col(ignition_df)
    hole_col_ignition = find_hole_size_col(
        ignition_df, exclude_cols={id_col_ignition, "P_jet", "P_flash", "P_vce", "P_pool"}
    )
    if hole_col_ignition is None:
        raise KeyError("Ignition_Probability 시트에서 hole size(누출구경) 컬럼을 찾지 못했습니다.")
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

    # --- Occupancy (넓은 포맷: Personnel_Category x Area 시간비율) + 그린 Area 매칭 ---
    occ_records = load_occupancy_wide(occ_df)
    occ_by_area = {}
    for r in occ_records:
        occ_by_area.setdefault(_norm_name(r["area_name"]), []).append(
            (r["population"], r["time_fraction"], r["category"])
        )

    matched_areas = []
    for a in areas:
        recs = occ_by_area.get(_norm_name(a["name"]), [])
        matched_areas.append({**a, "occupancy": recs})
    if not any(a["occupancy"] for a in matched_areas):
        raise KeyError(
            "그린 Area 이름과 Occupancy 시트의 구역명이 하나도 일치하지 않습니다. "
            "이름을 Occupancy 시트의 컬럼명과 맞춰주세요 "
            f"(Occupancy 시트에 있는 구역명: {sorted({r['area_name'] for r in occ_records})})"
        )

    # --- 기대값(expected value) 방식 ---
    # 사고 시나리오 하나(=IS_ID x 홀크기 x 기상 x 풍향 x 사고분기)마다
    #   N_event = sum_구역 [ 인원수 x 그 구역 체류시간비율 x 겹침비율 x 치사확률 ]
    # 을 "하나의 소수값"으로 계산한다 (이항분포/합성곱 없이, 업계 표준 PLL 산식과 동일).
    # events: (is_id, N_event, weight) 목록. weight = 누출빈도 x 풍향확률 x 착화확률.
    events = []

    def expected_band_event(weight, is_id, center, r_severe, r_mild, bearing, mode, hazard_type,
                             level_severe, level_mild):
        if weight <= 0:
            return
        n_event = 0.0
        for a in matched_areas:
            if not a["occupancy"]:
                continue
            frac_sev, frac_mld = rect_band_fractions(
                a["cx"], a["cy"], a["w"], a["h"], a["angle"],
                center, r_severe, r_mild, bearing, mode, back_frac, cross_frac
            )
            if frac_sev <= 0 and frac_mld <= 0:
                continue
            a_norm = _norm_name(a["name"])
            v_severe = get_vuln_for_area(hazard_type, level_severe, a_norm)
            v_mild = get_vuln_for_area(hazard_type, level_mild, a_norm)
            for pop, timefrac, cat in a["occupancy"]:
                n_event += pop * timefrac * (frac_sev * v_severe + frac_mld * v_mild)
        if n_event > 0:
            events.append((is_id, n_event, weight))

    def expected_single_event(weight, is_id, center, r, bearing, mode, hazard_type, level):
        if weight <= 0 or r is None or r <= 0:
            return
        n_event = 0.0
        for a in matched_areas:
            if not a["occupancy"]:
                continue
            frac = rect_single_fraction(a["cx"], a["cy"], a["w"], a["h"], a["angle"],
                                         center, r, bearing, mode, back_frac, cross_frac)
            if frac <= 0:
                continue
            v = get_vuln_for_area(hazard_type, level, _norm_name(a["name"]))
            for pop, timefrac, cat in a["occupancy"]:
                n_event += pop * timefrac * frac * v
        if n_event > 0:
            events.append((is_id, n_event, weight))

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
                    expected_band_event(weight, is_id, center, d["Pool_37.5"], d["Pool_12.5"], bearing,
                                         "lobe", "Thermal (Jet/Pool)", "37.5", "12.5")
                if cls["has_jet"]:
                    weight_jet = freq * wind_p * ign["P_jet"]
                    expected_band_event(weight_jet, is_id, center, d["Jet_37.5"], d["Jet_12.5"], bearing,
                                         "lobe", "Thermal (Jet/Pool)", "37.5", "12.5")

                    weight_flash = freq * wind_p * ign["P_flash"]
                    expected_single_event(weight_flash, is_id, center, d["Flash_LFL"], bearing,
                                           "lobe", "Flash Fire", "5 (Methane LFL)")

                    weight_vce = freq * wind_p * ign["P_vce"]
                    expected_band_event(weight_vce, is_id, center, d["Exp_0.5"], d["Exp_0.3"], bearing,
                                         "circle", "Overpressure", "0.5", "0.3")

    if not events:
        return np.array([]), np.array([]), pd.DataFrame(), matched_areas, 0.0

    # --- PLL(Potential Loss of Life, 연간 기대 사망자수) = 모든 시나리오의 weight x N 합 ---
    # F-N curve 표시 범위(N>=1)와 무관하게, N<1인 시나리오도 반드시 포함해서 계산한다.
    pll_total = sum(weight * n_event for _, n_event, weight in events)

    contrib_by_isid = {}
    for is_id, n_event, weight in events:
        contrib_by_isid[is_id] = contrib_by_isid.get(is_id, 0.0) + weight * n_event

    # --- F(N 이상) 누적곡선: N 오름차순으로 정렬해, 뒤(큰 N)에서부터의 누적합(suffix sum)을 구한다 ---
    events_sorted = sorted(events, key=lambda e: e[1])
    n_values = np.array([e[1] for e in events_sorted])
    weights_arr = np.array([e[2] for e in events_sorted])
    f_cum = np.cumsum(weights_arr[::-1])[::-1]  # f_cum[i] = N_values[i] 이상인 모든 시나리오의 빈도 합

    event_summary = pd.DataFrame(
        [{"IS_ID": k, "PLL 기여 (기대 사망자수/yr)": v} for k, v in contrib_by_isid.items()]
    )
    if not event_summary.empty:
        event_summary = event_summary.sort_values(
            "PLL 기여 (기대 사망자수/yr)", ascending=False
        ).reset_index(drop=True)

    return n_values, f_cum, event_summary, matched_areas, pll_total


def render_fn_curve():
    required = ["IS_Coordinates", "Leak_Frequencies", "PHAST_Distances",
                "Wind rose(Meteorology)", "Vulnerability_Criteria", "Ignition_Probability", "Occupancy"]
    miss = missing_sheets(required)
    if miss:
        st.warning(f"이 분석에 필요한 시트가 없습니다: {miss}")
        return

    areas = render_area_drawer()
    st.markdown("---")

    if not areas:
        st.info("F-N Curve를 계산하려면 위에서 Area를 최소 1개 이상 그리고 이름을 붙여주세요.")
        return

    st.caption(
        "방법론(기대값 방식, 업계 표준): (IS_ID, 누출구경, 기상조건, 풍향, 사고분기) 조합별로 "
        "빈도(=누출빈도 x 풍향확률 x 착화확률)를 산정합니다. 각 Area 사각형이 그 사고의 심각/경미 영향권과 "
        "몇 % 겹치는지 면적비로 계산하고, 시나리오별 기대 사망자수 "
        "N = Σ_구역[인원수 x 그 구역 체류시간비율 x 겹침비율 x 치사확률] 을 산정합니다(소수 가능, 이항분포/"
        "합성곱 없이 CCPS·HSE 표준 방식과 동일). 과압(0.3/0.5bar)은 구역의 Building_Type(Area_Coordinates 시트)에 "
        "따라 Indoor_Vuln_ByBuildingType 시트의 건물유형별 실내 치사확률을 적용하고, 옥외 구역은 Outdoor 값을 "
        "그대로 씁니다. 모든 시나리오를 N 오름차순으로 정렬해 F(N명 이상 사망 누적빈도)를 계산하고, "
        "N≥1 구간만 그래프로 표시합니다(N<1인 시나리오는 PLL 계산에는 포함하되 그래프에서는 관례상 제외)."
    )

    with st.spinner("F-N Curve 계산 중..."):
        try:
            areas_json = json.dumps(areas, sort_keys=True)
            n_values_all, f_cum_all, event_summary, matched_areas, pll_total = compute_fn_curve(
                excel_bytes, back_frac, cross_frac, areas_json
            )
        except Exception as e:
            st.error(f"F-N Curve 계산 중 오류가 발생했습니다 (엑셀 컬럼명이 템플릿과 다를 수 있습니다): {e}")
            return

    n_matched = sum(1 for a in matched_areas if a["occupancy"])
    st.caption(f"Occupancy 시트와 이름이 일치해 인원이 연결된 Area: {n_matched} / {len(matched_areas)}개")
    if n_matched < len(matched_areas):
        unmatched = [a["name"] for a in matched_areas if not a["occupancy"]]
        st.warning(f"이름이 일치하지 않아 인원이 연결되지 않은 Area: {unmatched}")

    if len(n_values_all) == 0:
        st.warning("계산된 인명피해 이벤트가 없습니다 — Area 위치/이름 또는 Occupancy 데이터를 확인해주세요.")
        return

    st.write(f"**PLL (연간 기대 사망자수) = {pll_total:.3e} /yr**  (N<1인 시나리오도 모두 포함한 값)")

    # 그래프는 업계 관례에 따라 N≥1 구간만 표시 (N<1은 IRPA/LSIR가 담당하는 개인리스크 영역과 중복되므로 제외)
    disp_mask = n_values_all >= 1.0
    n_values = n_values_all[disp_mask]
    f_cum = f_cum_all[disp_mask]

    if len(n_values) == 0:
        st.warning("N≥1(1명 이상 기대 사망)인 시나리오가 없어 F-N curve를 그래프로 표시할 수 없습니다. "
                    f"(PLL={pll_total:.3e}/yr은 위에 표시된 값대로 유효합니다.)")
        return

    st.write(f"F(N≥1) = {f_cum[0]:.3e} /yr  |  최대 N = {n_values[-1]:.2f}명")

    show_markers = st.checkbox("계산 지점에 마커 표시", value=False, key="fn_show_markers")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.step(n_values, f_cum, where="post", color="tab:blue", linewidth=2,
            marker="o" if show_markers else None, markersize=3)

    x_lo, x_hi = 1, max(n_values[-1], 2)
    n_line = np.logspace(math.log10(x_lo), math.log10(x_hi), 100)
    ax.plot(n_line, 1e-2 / n_line, "r--", label="Unacceptable (F=1e-2/N)")
    ax.plot(n_line, 1e-4 / n_line, "g--", label="Acceptable (F=1e-4/N)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    finite_f = f_cum[f_cum > 0]
    if len(finite_f):
        ax.set_ylim(finite_f.min() * 0.3, max(f_cum.max() * 3, 1e-2))
    ax.set_xlabel("N (Expected Number of Fatalities)")
    ax.set_ylabel("F (Cumulative Frequency of N or more fatalities, /yr)")
    ax.set_title("F-N Curve (expected-value method)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    st.pyplot(fig, **PYPLOT_WIDTH_KW)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button("결과 이미지 다운로드 (PNG)", buf, "fn_curve.png", "image/png", key="dl_fn_png")
    plt.close(fig)

    fn_table = pd.DataFrame({"N": n_values_all, "F_cumulative_per_yr": f_cum_all})
    csv_buf = fn_table.to_csv(index=False).encode("utf-8-sig")
    st.caption("CSV에는 그래프에 표시되지 않는 N<1 시나리오도 모두 포함되어 있습니다 (PLL 검산용).")
    st.download_button("F-N 데이터 다운로드 (CSV, N<1 포함)", csv_buf, "fn_curve_data.csv", "text/csv", key="dl_fn_csv")

    if event_summary is not None and not event_summary.empty:
        st.markdown("**IS_ID별 PLL 기여도 (내림차순)**")
        st.dataframe(event_summary, **DATAFRAME_WIDTH_KW)


# ------------------------------------------------------------------
# 탭 렌더링
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["💥 Explosion (Overpressure)", "🔥 Thermal Radiation", "☠️ LSIR", "📊 F-N Curve"]
)

with tab1:
    render_explosion()

with tab2:
    render_thermal()

with tab3:
    render_lsir()

with tab4:
    render_fn_curve()
