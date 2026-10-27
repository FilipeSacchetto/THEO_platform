# THEO — graph-informed time-series forecasting

Predict the future of any set of things that change over time — energy use,
weather, sales, traffic, and so on. Upload **any** table that records several
items across time, point THEO at the number you want to forecast, choose a
model, and press **Run**. You get accuracy numbers plus a table and a plot of
what was predicted.

You do **not** need to write code to use it. Open the visual interface in your
browser and follow the steps below. (Developers who want to call the engine
directly from Python will find that at the bottom of this page.)

THEO works by looking at your items **together**. If some of your items are
near each other (by location, or simply by behaving similarly), it uses the
relationship between them to improve the forecast. This is the "graph" in
THEO — you never build it by hand, the tool does it automatically.

---

## How to use THEO (step by step)

### Before you start — install the tools once

The interface runs on a small, free Python library called **Streamlit**. You
only do this **once**, in a terminal.

1. Open a **terminal** (on Mac: press `Command + Space`, type "Terminal",
   press Enter).
2. Install the required programs (this may take a few minutes):

   ```bash
   cd /Users/filipesacchetto/Documents/GitHub/Mestrado/THEO_platform
   python -m venv theo_env
   source theo_env/bin/activate
   pip install -r requirements.txt
   ```

   > If you prefer conda instead, you can create and activate your conda
   > environment instead of the `python -m venv` lines above.

### Step 1 — launch the interface

With the terminal still open (and the environment activated), run:

```bash
cd code_to_deploy
streamlit run app.py
```

A page opens automatically in your browser (usually
`http://localhost:8501`). If it doesn't open, just visit that link by hand.

You should see a screen titled **"THEO"** with a section called **"1 · Data"**.

### Step 2 — bring in your data

At the top of the page (**1 · Data**):

- Click **"Upload a file"** and choose a **CSV** file, **or**
- Type (or paste) the **path to a CSV file** in the box next to it.

Your table should look like this: one row for each *item* on each *moment in
time*. For example, rows like *(solar plant A, Jan 2020), (solar plant A, Feb
2020), (solar plant B, Jan 2020)…*.

Just below, you'll see a small preview of the first rows of your file.

### Step 3 — tell THEO which columns mean what

Scroll down to **"2 · Mapping, model & run"**. You'll see several drop-down
lists. The tool guesses the answers for you — but please double-check each one.

| Field | What it means |
|-------|---------------|
| **Entity / node id** | The column that names *which item* each row is (e.g. the plant's name, the store's ID). |
| **Time column** | The column with the date/period (e.g. month or year). |
| **Target (what to predict)** | The number you want to forecast (e.g. energy production). |
| **Year column** | If your time column is just months, add the year here so THEO orders everything correctly. |

### Step 4 — let THEO learn how your items relate

There are two ways to describe the connection between your items. The tool
works out which one you're using automatically, based on what you fill in.
**You only need to do one of them.**

**Option A — "Build it from a clean table" (most common)**

If your data has real measurements for each item, use this. The tool builds the
graph on its own.

- **Position X (e.g. lon)** — the column with each item's horizontal location
  (like longitude).
- **Position Y (e.g. lat)** — the column with each item's vertical location
  (like latitude).
- **THEO variables** — select **all** the measurement columns you have for each
  item. These become the "features" the model reads. You can include the target
  itself.

**Option B — "Consume a pre-built THEO table"**

Use this only if your CSV already contains special columns named with `_node`
and `_edge`.

- **Node features (`_node`)** — select the `_node` columns.
- **Edge features (`_edge`)** — select the `_edge` columns.

> **Note:** to build the graph from positions (Option A) your file must contain
> the **X** and **Y** location columns. If you're using a table that already has
> `_node`/`_edge` columns (Option B), those position columns aren't needed.

### Step 5 — choose the model and settings

Still in section 2:

- **Window size** — how many past time steps the model looks back on before
  making a prediction. *24 is a common starting point; if your data repeats
  yearly, 12 works for monthly data, 365 for daily data.*
- **Model** — pick one of the three:
  - **LSTM** — a strong general-purpose model; a good default.
  - **ANN** — a simpler, faster model.
  - **GNN** — uses the graph most directly; **needs** either location columns
    (Option A) or a neighbour column.
- **Epochs** — how many times the model practices while learning. *10 is a
  fine starting point; more can help but slows things down.*
- **Seed** — a number that makes results reproducible. Leave it as **42**.

Click the **"Split & advanced"** triangle if you want to change how the data is
divided between learning, tuning, and testing, or choose a specific computer
chip ("Device"). Leave these alone unless you know what they do.

### Step 6 — run it

Click **"Run"** at the bottom of the section.

A green progress bar appears at the top and moves through the stages:
aggregating the data → building the window → training → evaluating.

> If something goes wrong, a red message explains what, and you can expand
> **"Full traceback"** for the technical details.

### Step 7 — read your results

You'll land on the **"Results"** section:

- **Four summary numbers** at the top measure accuracy:
  - **RMSE** and **MAE** — how far off the predictions are on average (lower
    is better).
  - **R²** — a score from 1.0 downwards (closer to 1.0 is better).
- **A predictions table** with every *entity*, its *time key*, the *actual*
  value, and the *predicted* value. You can download it with the
  **"Download predictions (CSV)"** button.
- **A plot** showing actual vs. predicted for one sample item, so you can see
  how well the forecast tracks reality.

That's it — you've forecast with THEO! 🎉

---

## For developers (Python)

You can also run THEO directly from Python instead of using the visual
interface.

**Build THEO from a clean table** (Delaunay triangulation over positions):

```python
from theo import ColumnSpec, TrainConfig, SplitConfig, run_pipeline
from theo.data.loader import load_table

df = load_table("databases/raw_databases/preprocessed_dfsfull_sensors_brasil.csv")
spec = ColumnSpec(
    entity="id", time="month", year="year",
    target="ALLSKY_SFC_SW_DWN",                          # the feature to predict
    theo_variables=["ALLSKY_SFC_SW_DWN", "PS", "RH2M", "T2M", "WS10M"],
    x_col="lon", y_col="lat",                            # Delaunay positions
)
result = run_pipeline(df, spec, model="ann",
                      train_cfg=TrainConfig(window_size=24, epochs=10),
                      split_cfg=SplitConfig(train_frac=0.70, val_frac=0.85))
print(result["test"])          # {'rmse': …, 'mae': …, 'r2': …}
```

**Consume a pre-built `_node`/`_edge` table:**

```python
from theo import ColumnSpec, TrainConfig, SplitConfig, run_pipeline
from theo.data.loader import load_table

df = load_table("databases/preprocessed_dfs/knn5_sp_sensorspreprocessed_df_full.csv")
spec = ColumnSpec(
    entity="source",
    time="month",
    target="radiation_node",
    node_features=["radiation_node", "temperature_node", "surface_pressure_node",
                   "relative_hum_node", "wind_speed_node"],
    edge_features=["radiation_edge", "temperature_edge", "surface_pressure_edge",
                   "relative_hum_edge", "wind_speed_edge"],
    neighbor="target",
    year="year",
)
result = run_pipeline(df, spec, model="lstm",
                      train_cfg=TrainConfig(window_size=24),
                      split_cfg=SplitConfig(train_frac=0.70, val_frac=0.85))
print(result["test"])          # {'rmse': …, 'mae': …, 'r2': …}
print(len(result["predictions"]))  # n (entity, month) test pairs
```

---

## What THEO does under the hood

THEO is **data-agnostic**: it doesn't assume anything about your field (it's
not locked to solar irradiance). You supply any table that records several
items over time, map the columns, choose the **target** and the **window
size**, and pick a model. Every model uses the same THEO feature engineering —
each item's own values (**node features**) plus the differences between each
item and its neighbours (**edge features**). Only the way the window/graph is
read changes between models.

The graph can be **built** from a clean table (a Delaunay triangulation of each
item's 2D position) or **consumed** from a pre-built `_node`/`_edge` table.
Both paths feed the models the same fixed-name feature set.

### Models

| Model | What it reads | Role in THEO |
|-------|---------------|--------------|
| **ANN**  | Flattened window (node + edge features) | Edge features injected as inputs |
| **LSTM** | Temporal window (node + edge features) | Edge features injected as inputs |
| **GNN**  | Per-step graph (node features + edges) | Neighbour aggregation learned |

All three share identical data, target, window, split, architecture, epochs
and seed — the only difference is the model itself.

### Logging

Every run is written to a timestamped file under `logs/` (and echoed to the
terminal): data size, column mapping, split points, window/model counts,
per-epoch validation loss, final metrics, and the full traceback of any
failure. The log file path is shown at the bottom of the page.

### Upload size

`.streamlit/config.toml` raises the upload limit to 50 GB, so the default
200 MB cap won't constrain your data.
