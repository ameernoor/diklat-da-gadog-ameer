import pickle
from pathlib import Path

import altair as alt
import pandas as pd
import Orange
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard Analitik DJPb", layout="wide")
st.title("Dashboard Analitik DJPb")
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
	return pd.read_csv(path)

@st.cache_data
def load_model(path: Path):
	with open(path, "rb") as f:
		return pickle.load(f)

base_path = Path(__file__).resolve().parent
model_path = base_path / "model" / "Best_model.pkcls"
data_path = base_path / "data" / "02_realisasi_anggaran_klasifikasi.csv"

model = load_model(model_path)
data = load_data(data_path)

numeric_columns = data.select_dtypes(include="number").columns.tolist()
categorical_columns = [
	col for col in data.columns if data[col].dtype == "object" or data[col].dtype.name == "category"
]

st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

st.sidebar.markdown("---")
st.sidebar.header("Data Preview")
start_row = st.sidebar.slider(
	"Pilih baris awal", 1, max(1, len(data) - 9), 1, help="Tampilkan 10 baris data mulai dari baris ini"
)
preview = data.iloc[start_row - 1 : start_row - 1 + 10]
st.subheader(f"Preview data (baris {start_row} sampai {start_row + len(preview) - 1})")
st.dataframe(preview)
st.subheader("Scatter Plot Dinamis")
st.sidebar.markdown("---")
st.sidebar.header("Scatter Plot Controls")
scatter_x = st.sidebar.selectbox("X feature", numeric_columns, index=0)
scatter_y = st.sidebar.selectbox("Y feature", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
color_feature = st.sidebar.selectbox(
	"Color by (categorical)", categorical_columns, index=0
) if categorical_columns else None
if scatter_x and scatter_y:
	x_min, x_max = float(data[scatter_x].min()), float(data[scatter_x].max())
	y_min, y_max = float(data[scatter_y].min()), float(data[scatter_y].max())
	x_range = st.sidebar.slider("X axis range", x_min, x_max, (x_min, x_max))
	y_range = st.sidebar.slider("Y axis range", y_min, y_max, (y_min, y_max))

	scatter_data = data[
		data[scatter_x].between(x_range[0], x_range[1])
		& data[scatter_y].between(y_range[0], y_range[1])
	]

	scatter = (
		alt.Chart(scatter_data)
		.mark_circle(size=60, opacity=0.7)
		.encode(
			x=alt.X(scatter_x, scale=alt.Scale(domain=x_range)),
			y=alt.Y(scatter_y, scale=alt.Scale(domain=y_range)),
			color=color_feature if color_feature else alt.value("steelblue"),
			tooltip=[*numeric_columns, *categorical_columns],
		)
		.properties(height=450, width=700)
		.interactive()
	)
	st.altair_chart(scatter, use_container_width=True)

st.markdown("---")

def build_prediction_input(selected_category: str) -> pd.DataFrame:
	category_keys = [attr.name for attr in model.domain.attributes if attr.name.startswith("tipe_satker=")]
	base_row = {
		attr.name: 0
		for attr in model.domain.attributes
		if attr.name.startswith("tipe_satker=")
	}
	base_row.update(
		{
			"jumlah_spm": st.number_input(
				"Jumlah SPM",
				value=int(data["jumlah_spm"].median()),
				min_value=int(data["jumlah_spm"].min()),
				max_value=int(data["jumlah_spm"].max()),
			),
			"revisi_dipa": st.number_input(
				"Revisi DIPA",
				value=int(data["revisi_dipa"].median()),
				min_value=int(data["revisi_dipa"].min()),
				max_value=int(data["revisi_dipa"].max()),
			),
			"deviasi_rpd_persen": st.number_input(
				"Deviasi RPD (%)",
				value=float(data["deviasi_rpd_persen"].median()),
				min_value=float(data["deviasi_rpd_persen"].min()),
				max_value=float(data["deviasi_rpd_persen"].max()),
				step=0.1,
			),
			"skor_ikpa": st.number_input(
				"Skor IKPA",
				value=float(data["skor_ikpa"].median()),
				min_value=float(data["skor_ikpa"].min()),
				max_value=float(data["skor_ikpa"].max()),
				step=0.1,
			),
			f"tipe_satker={selected_category}": 1,
		}
	)
	return pd.DataFrame([base_row])

st.header("Prediksi")
if opsi == "Regresi":
	st.warning("Model regresi belum tersedia. Pilih Klasifikasi untuk menggunakan Best_model.pkcls.")
else:
	st.info("Masukkan nilai fitur untuk melakukan prediksi klasifikasi.")
	satker_options = sorted(data["tipe_satker"].dropna().unique().tolist())
	selected_satker = st.selectbox("Tipe Satker", satker_options)
	prediction_df = build_prediction_input(selected_satker)
	st.subheader("Input fitur model")
	st.dataframe(prediction_df)

	if st.button("Jalankan Prediksi Klasifikasi"):
		prediction_domain = Orange.data.Domain(model.domain.attributes)
		table = Orange.data.Table(prediction_domain, prediction_df)
		prediction_result = model.predict(table)

		if isinstance(prediction_result, tuple):
			labels, probs = prediction_result
			label_index = int(labels[0])
			predicted_class = model.domain.class_var.values[label_index]
			prob_df = pd.DataFrame(
				[probs[0]],
				columns=model.domain.class_var.values,
			)
		else:
			label_index = int(prediction_result[0])
			predicted_class = model.domain.class_var.values[label_index]
			prob_df = None

		st.success(f"Prediksi: {predicted_class}")
		if prob_df is not None:
			st.subheader("Probabilitas kelas")
			st.dataframe(prob_df)