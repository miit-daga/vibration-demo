# %% [markdown]
# DRDO motor bearing diagnosis, demo app
#
#   streamlit run app.py
#
# CSV in, analysis graphs, fault type out. That is the whole brief.
#
# The feature extraction below is copied verbatim from motor_eda.py. It has to
# be: the model was trained on those exact numbers, and a subtly different
# implementation here would not error, it would just produce confident wrong
# answers. There is a self-check in the sidebar that recomputes features for
# real training windows and compares them against the values in the EDA
# parquet, so a drift between the two files is caught rather than shipped.
import io
import os
import hmac
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1  # noqa: F401  (st.components.v1)
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy import fft as sp_fft

st.set_page_config(page_title="Bearing Fault Diagnosis", layout="wide")


def setting(name, default=None):
    """A configuration value, from Streamlit secrets or from the environment.

    Community Cloud does expose top-level secrets as environment variables, so
    os.environ alone would usually be enough there, but only for top-level
    entries and only on that host. Checking st.secrets first means one code path
    works whether the value arrives as a Community Cloud secret, a container
    environment variable, or a local .streamlit/secrets.toml."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # No secrets file present, which is the normal case locally and not an
        # error. The exception type has changed across Streamlit versions, hence
        # the broad catch rather than naming one.
        pass
    return os.environ.get(name, default)


def require_password():
    """Gate the page behind a shared password whenever APP_PASSWORD is set.

    Letting someone open the app without a Hugging Face or GitHub account means
    the app has to be reachable at a public URL, so the URL itself cannot be the
    secret. This keeps the page reachable but useless without the password. It
    runs before the artifacts are loaded and before anything else is drawn, so an
    uninvited visitor sees a bare sign-in and learns nothing about the project,
    not even from the sidebar. Leave the variable unset locally and the gate
    disappears entirely.

    Not a substitute for real access control: it is one shared secret over TLS,
    which is proportionate for a demonstration and nothing more."""
    want = setting("APP_PASSWORD")
    if not want or st.session_state.get("authed"):
        return
    st.title("Sign in")
    with st.form("auth"):
        got = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            # compare_digest rather than ==, so a wrong guess takes the same time
            # to reject whatever it got right.
            if hmac.compare_digest(got, want):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()


require_password()


@st.cache_resource
def artifact_dir():
    """Where the model files are read from.

    Normally the directory in this repo. If HF_ARTIFACT_REPO is set the files are
    pulled from a private Hugging Face dataset instead, authenticated with the
    HF_TOKEN secret. That combination gives a public app -- anyone with the link
    opens it, no login -- while the model and the recording excerpts stay in a
    private repo rather than being browsable and search-indexable. Cached, so the
    download happens once per process rather than on every Streamlit rerun."""
    local = setting("ART_DIR", "model_out")
    repo = setting("HF_ARTIFACT_REPO")
    if not repo:
        return local
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, repo_type="dataset",
                             token=setting("HF_TOKEN"))


ART = artifact_dir()
EDA = os.environ.get("EDA_DIR", "eda_out")
PARQ = os.environ.get("PARQUET_DIR", "parquet_cache")

CLASS_COLOR = {"Healthy": "#0072B2", "Ball Fault": "#E69F00",
               "Inner Race Fault": "#009E73", "Outer Race Fault": "#D55E00",
               "Combined Fault": "#CC79A7"}
DEFECT_COLOR = {"BPFO": "#D55E00", "BPFI": "#009E73", "BSF": "#E69F00",
                "FTF": "#CC79A7"}
# The figures are drawn on a transparent background so they sit on the page
# rather than in a white box. That choice means every bit of foreground ink has
# to be set explicitly, because matplotlib knows nothing about the Streamlit
# theme: on the dark theme its default near-black text, ticks and spines are
# invisible against the page, which is exactly what happened. st.context.theme
# reports the theme the viewer is actually seeing, including when that came from
# their operating system rather than a config file.
try:
    DARK = st.context.theme.type == "dark"
except (AttributeError, KeyError, TypeError):
    DARK = False
FG = "#e6e6e6" if DARK else "#111111"     # axes, labels, and "your recording"
MUTED = "#9a9a9a" if DARK else "#666666"  # grid, and unknown-class fallback
BG = "#0e1117" if DARK else "#ffffff"     # marker edges, to punch a visible hole

mpl.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "none", "axes.facecolor": "none",
                     "legend.frameon": False,
                     "text.color": FG, "axes.labelcolor": FG,
                     "axes.edgecolor": FG, "xtick.color": FG,
                     "ytick.color": FG, "grid.color": MUTED,
                     "legend.labelcolor": FG})


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def pickled_sklearn_version(path):
    """The version a scikit-learn pickle was written with, read out of the raw
    bytes. Unpickling across versions fails with messages like 'No module named
    _loss' that name neither scikit-learn nor a version, so this exists to turn
    that into something actionable."""
    import re
    try:
        b = open(path, "rb").read(400000)
        m = re.search(rb"_sklearn_version.{0,20}?([0-9]+\.[0-9]+\.[0-9]+)",
                      b, re.S)
        if m:
            return m.group(1).decode()
        v = re.findall(rb"[0-9]+\.[0-9]+\.[0-9]+", b)
        return v[0].decode() if v else None
    except OSError:
        return None


@st.cache_resource
def load_artifacts():
    with open(f"{ART}/40_production_model.pkl", "rb") as fh:
        M = pickle.load(fh)
    C = json.load(open(f"{ART}/41_serving_config.json"))
    R = np.load(f"{ART}/42_reference_curves.npz")
    FR_ = pd.read_csv(f"{ART}/43_feature_reference.csv")
    IMP = pd.read_csv(f"{ART}/44_feature_importance.csv")
    return M, C, {k: R[k] for k in R.files}, FR_, IMP


try:
    MODEL, CFG, REF, FEATREF, IMPORT = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Cannot find the model artifacts.\n\n`{e}`\n\nRun cell 13 of "
             f"`motor_model.py`, then point `ART_DIR` at its output directory. "
             f"Currently looking in `{os.path.abspath(ART)}`.")
    st.stop()
except (ModuleNotFoundError, AttributeError, ImportError) as e:
    import sklearn
    want = pickled_sklearn_version(f"{ART}/40_production_model.pkl")
    st.error(
        f"**scikit-learn version mismatch.** The model cannot be unpickled.\n\n"
        f"`{type(e).__name__}: {e}`\n\n"
        f"The model was saved with scikit-learn **{want or 'unknown'}**; this "
        f"environment has **{sklearn.__version__}**. Pickles of tree ensembles "
        f"are not portable across versions.\n\n"
        f"```\npip install scikit-learn=={want or '1.6.1'}\n```")
    st.stop()

FS = CFG["fs_hz"]
FR = CFG["shaft_rate_hz"]
WLEN = CFG["window_samples"]
OVERLAP = CFG["overlap"]
BAND_LO, BAND_HI = CFG["demod_band_hz"]
AXES = CFG["axes"]
CLASSES = CFG["classes"]
FEATURE_COLS = CFG["feature_cols"]
DEFECT_HZ = CFG["defect_hz"]
DEFECT_ORDERS = CFG["defect_orders"]
FMAX_ENV = CFG["fmax_env_hz"]


def alarm_threshold():
    """The health-index alarm level, derived from the saved out-of-fold scores
    rather than taken from the config.

    The exported `alarm_threshold` is the 1 percent false-alarm operating point,
    and on this data that percentile is set by the top one or two of 154 healthy
    windows: the 99th percentile is 18,014 while the 98th is 450, a fortyfold
    step for one percentile of very little data. At 18,014 the alarm catches only
    76.6 percent of fault windows, and just 4.7 percent of Combined Fault -- the
    most severe condition in the set -- because that recording's saturated x
    channel is excluded from the amplitude-invariant feature set, so the detector
    sees it through its quieter axes and it scores lower than the single faults.

    Used here instead: the lowest false-alarm rate that reaches full detection on
    the reference scores, which is 2.6 percent and matches the detection figure
    reported in the results. Derived rather than hardcoded, so it tracks the model
    if it is ever retrained. Falls back to the config value if the reference
    scores are missing."""
    hs, lb = REF.get("health_scores"), REF.get("health_labels")
    if hs is None or lb is None:
        return float(CFG["alarm_threshold"]), None
    ok = np.isfinite(hs)
    hs, lb = hs[ok], lb[ok].astype(str)
    he, fa = hs[lb == "Healthy"], hs[lb != "Healthy"]
    if len(he) < 20 or len(fa) < 20:
        return float(CFG["alarm_threshold"]), None
    for p in np.arange(99.0, 90.0, -0.5):
        t = float(np.percentile(he, p))
        if (fa > t).mean() >= 1.0:
            return t, float((he > t).mean())
    return float(CFG["alarm_threshold"]), None


THRESH, THRESH_FAR = alarm_threshold()
OOD = CFG["ood_bound"]
HOP = int(WLEN * (1 - OVERLAP))
FEAT_DEFECTS = dict(DEFECT_HZ)


# ---------------------------------------------------------------------------
# Feature extraction, copied verbatim from motor_eda.py. Do not "improve" this.
# ---------------------------------------------------------------------------
def bandpass(v, lo, hi, fs, order=4):
    lo = max(lo, 1.0)
    hi = min(hi, fs / 2 - 1.0)
    if hi <= lo:
        return None
    sos = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band",
                        output="sos")
    return signal.sosfiltfilt(sos, v)


def envelope(v):
    n = len(v)
    e = np.abs(signal.hilbert(v, N=sp_fft.next_fast_len(n))[:n])
    return e - e.mean()


def envelope_spectrum(v, fs, lo, hi, nperseg):
    b = bandpass(v - v.mean(), lo, hi, fs)
    if b is None:
        return None, None
    e = envelope(b)
    f, P = signal.welch(e, fs=fs, nperseg=min(nperseg, len(e)),
                        noverlap=min(nperseg, len(e)) // 2, detrend="constant")
    return f, P


def peak_over_floor(f, P, target, tol=0.004, floor_bw=15.0):
    amp = np.sqrt(P)
    tol_hz = max(2.0 * (f[1] - f[0]), tol * target)
    near = np.abs(f - target) <= tol_hz
    if near.sum() == 0:
        return np.nan, np.nan
    peak = float(np.max(amp[near]))
    ring = (np.abs(f - target) <= floor_bw) & ~near
    floor = float(np.median(amp[ring])) if ring.sum() > 0 else np.nan
    return peak, (peak / floor if floor and floor > 0 else np.nan)


def window_features(v, fs, lo, hi):
    v = v - v.mean()
    rms = float(np.std(v, ddof=0))
    peak = float(np.max(np.abs(v)))
    absmean = float(np.mean(np.abs(v)))
    feats = {
        "rms": rms, "peak": peak,
        "kurtosis": float(stats.kurtosis(v, fisher=True, bias=False)),
        "skew": float(stats.skew(v, bias=False)),
        "crest": peak / (rms + 1e-20),
        "impulse": peak / (absmean + 1e-20),
        "shape": rms / (absmean + 1e-20),
        "clearance": peak / (np.mean(np.sqrt(np.abs(v))) ** 2 + 1e-20),
        "p2p": float(np.ptp(v)),
    }
    f, P = signal.welch(v, fs=fs, nperseg=len(v), detrend="constant")
    tot = np.sum(P) + 1e-20
    for a_lo, a_hi in [(0, 100), (100, 300), (300, 600),
                       (600, 1200), (1200, 2000), (2000, fs / 2)]:
        m = (f >= a_lo) & (f < a_hi)
        feats[f"bandfrac_{a_lo}_{int(a_hi)}"] = float(np.sum(P[m]) / tot)
    pn = P / tot
    feats["spec_entropy"] = float(-np.sum(pn * np.log(pn + 1e-20)))
    feats["spec_centroid"] = float(np.sum(f * P) / tot)
    for h in [1, 2, 3]:
        feats[f"shaft_{h}x"] = peak_over_floor(f, P, h * FR)[1]

    fe, Pe = envelope_spectrum(v, fs, lo, hi, len(v))
    if fe is not None:
        for name, hz in FEAT_DEFECTS.items():
            for h in [1, 2, 3]:
                feats[f"env_{name}_h{h}"] = peak_over_floor(fe, Pe, h * hz)[1]
        for name, sb, tag in [("BPFI", FR, "fr"),
                              ("BSF", DEFECT_HZ["FTF"], "ftf")]:
            c = FEAT_DEFECTS[name]
            centre = peak_over_floor(fe, Pe, c)[0]
            lsb = peak_over_floor(fe, Pe, max(c - sb, 1.0))[0]
            usb = peak_over_floor(fe, Pe, c + sb)[0]
            feats[f"sbr_{name}_{tag}"] = float((lsb + usb) / (centre + 1e-20))
        feats["env_kurt"] = float(stats.kurtosis(
            envelope(bandpass(v, lo, hi, fs)), fisher=True, bias=False))
    return feats


def features_for_window(w):
    """w is (3, WLEN). Returns the full named feature dict for all four
    channels, matching the column names the model was trained on."""
    rec = {}
    for a_i, a in enumerate(AXES):
        for k, val in window_features(w[a_i].astype(np.float64), FS,
                                      BAND_LO, BAND_HI).items():
            rec[f"{a}_{k}"] = val
    mag = np.linalg.norm(w.astype(np.float64), axis=0)
    for k, val in window_features(mag, FS, BAND_LO, BAND_HI).items():
        rec[f"mag_{k}"] = val
    return rec


def predict(windows):
    """windows is (n, 3, WLEN)."""
    rows = [features_for_window(w) for w in windows]
    X = pd.DataFrame(rows).reindex(columns=FEATURE_COLS)
    Xv = np.nan_to_num(X.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    proba = MODEL["classifier"].predict_proba(Xv)
    health = MODEL["health_cov"].mahalanobis(
        MODEL["health_scaler"].transform(Xv))
    return X, Xv, proba, health


# ---------------------------------------------------------------------------
# CSV parsing. Their exports carry a units row under the header, which
# to_numeric turns into all-NaN, and one NaN is enough to poison everything.
# ---------------------------------------------------------------------------
def read_csv_any(buf):
    raw = pd.read_csv(buf, header=None, nrows=5)
    first_numeric = pd.to_numeric(raw.iloc[0], errors="coerce").notna().all()
    buf.seek(0)
    df = pd.read_csv(buf, header=None if first_numeric else 0)
    if df.shape[1] < 3:
        raise ValueError(f"Need at least 3 columns (x, y, z), found "
                         f"{df.shape[1]}.")
    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ["t"] + AXES
    else:
        df = df.iloc[:, :3]
        df.columns = AXES
    df = df.apply(pd.to_numeric, errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=AXES).reset_index(drop=True)
    return df, n_before - len(df)


def to_windows(df):
    arr = df[AXES].to_numpy()
    idx = np.arange(0, max(1, len(arr) - WLEN + 1), HOP)
    return np.stack([arr[i:i + WLEN].T for i in idx]), idx / FS


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
# The presenter's brief ships alongside the model in the private artifact store,
# not in this repository, because the repository is public so that Community
# Cloud can deploy it. Serving it here means one URL and one password cover both
# the demonstration and the document, rather than the brief needing its own host
# and its own access control.
BRIEF = f"{ART}/46_briefing.html"
if os.path.exists(BRIEF):
    page = st.sidebar.radio("Page", ["Demonstration", "Briefing"],
                            label_visibility="collapsed")
    if page == "Briefing":
        doc = open(BRIEF, encoding="utf-8").read()
        st.sidebar.download_button(
            "Download to read full screen", doc, mime="text/html",
            file_name="bearing-diagnosis-briefing.html")
        st.sidebar.caption("The document is rendered in a frame below, so it "
                           "scrolls inside the page. Downloading it and opening "
                           "the file in a browser tab reads better.")
        # An iframe rather than st.html: the document carries its own complete
        # stylesheet, including body rules, which would otherwise leak out and
        # restyle Streamlit's own chrome. height="content" lets the frame grow to
        # the document so the page scrolls once instead of nesting a scrollbar.
        # st.components.v1.html is the fallback for Streamlit older than st.iframe,
        # and is itself past its announced removal date, so it is not the default.
        if hasattr(st, "iframe"):
            st.iframe(Path(BRIEF), height="content")
        else:
            st.components.v1.html(doc, height=1700, scrolling=True)
        st.stop()

st.sidebar.title("Input")
st.sidebar.caption(
    f"CSV with columns t, x, y, z (or just x, y, z) sampled at {FS:.0f} Hz. "
    f"At least **{WLEN:,} rows** ({CFG['window_sec']:.0f} s), because the "
    f"defect frequencies are {min(DEFECT_HZ.values()):.0f} to "
    f"{max(DEFECT_HZ.values()):.0f} Hz and need a window to resolve. A single "
    f"x, y, z reading cannot be classified."
)
@st.cache_resource
def load_demo_recordings():
    """Short excerpts of the five recordings, bundled with the model by
    export_demo_data.py so a demonstration needs no external files. Optional:
    without it the app still works, it just loses the one-click examples and the
    sample CSV downloads, and falls back to PARQUET_DIR."""
    p = f"{ART}/45_demo_recordings.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {"classes": [str(c) for c in z["classes"]], "data": z["data"],
            "start_sec": z["start_sec"], "fs": float(z["fs"])}


DEMO = load_demo_recordings()


def demo_frame(k):
    a = DEMO["data"][k]
    return pd.DataFrame({"t": np.arange(len(a)) / DEMO["fs"],
                         **{ax: a[:, i] for i, ax in enumerate(AXES)}})


@st.cache_data
def demo_csv(k):
    return demo_frame(k).to_csv(index=False).encode()


up = st.sidebar.file_uploader("Upload recording", type=["csv", "txt"])

example = None
if not up:
    picks = ["(none)"] + (DEMO["classes"] if DEMO else CLASSES)
    example = st.sidebar.selectbox(
        "or load a known recording to see how it works", picks)
    if example == "(none)":
        example = None

# Downloading a sample and uploading it back is the only way to demonstrate the
# CSV path itself rather than the shortcut, which matters when the brief was a
# CSV upload. It also means a presenter with no files on their machine can still
# show the whole flow.
if DEMO is not None:
    with st.sidebar.expander("Download a sample CSV"):
        st.caption("A 16 second excerpt of each recording, in the same format "
                   "the uploader expects. Download one and upload it back to "
                   "demonstrate the upload path end to end.")
        for k, c in enumerate(DEMO["classes"]):
            st.download_button(c, demo_csv(k), key=f"dl{k}", mime="text/csv",
                               file_name=f"{c.replace(' ', '_')}.csv")

with st.sidebar.expander("Model card"):
    st.write(f"**Cross-validated macro F1** {CFG['cv_macro_f1']:.4f} "
             f"[{CFG['cv_macro_f1_ci95'][0]:.4f}, "
             f"{CFG['cv_macro_f1_ci95'][1]:.4f}]")
    st.write(f"**Bearing** {CFG['bearing']['name']}")
    st.write(f"**Shaft** {CFG['shaft_rpm']:.0f} RPM ({FR:.2f} Hz)")
    st.write(f"**Demodulation band** {BAND_LO:.0f} to {BAND_HI:.0f} Hz on "
             f"axis {CFG['envelope_axis']}")
    st.write("**Defect frequencies**")
    st.dataframe(pd.DataFrame(
        [{"line": k, "orders": round(DEFECT_ORDERS[k], 3),
          "Hz": round(DEFECT_HZ[k], 2)} for k in ["FTF", "BSF", "BPFO", "BPFI"]],
    ), hide_index=True, width="stretch")
    if THRESH_FAR is not None:
        st.write(f"**Health index alarm** {THRESH:,.0f}, the lowest false alarm "
                 f"rate reaching 100% detection on held-out windows "
                 f"({THRESH_FAR:.1%})")
    st.warning(CFG["scope_limits"])

# Feature extraction here is a verbatim copy of motor_eda.py, and a drift between
# the two would not raise an error. It would compute slightly different numbers,
# hand them to a model expecting the originals, and return confident wrong
# answers. That is the one failure mode with no symptoms, so it gets a test.
#
# The primary test runs anywhere, including on a host that has nothing but the
# model and the bundled excerpts: classify all five known recordings and check
# that each comes back as itself. It exercises the entire chain -- windowing,
# feature extraction, scaling, the classifier and the health index -- against
# data whose correct answer is known, so drifted feature code surfaces as a
# misclassification. Results go to the main area rather than the sidebar, because
# a narrow column is the wrong place for something an audience is meant to read.
#
# The second test is stricter but needs the EDA feature table and the full
# recordings, so it only appears on a machine that has them.
STRICT_TABLE = f"{EDA}/10_window_features.parquet"

with st.sidebar.expander("Self-check: is the model behaving?"):
    if DEMO is not None:
        st.caption("Classifies all five known recordings and checks each comes "
                   "back as itself. Covers everything from windowing to the "
                   "final verdict, on data whose answer is known.")
        if st.button("Run self-check", key="sc_e2e"):
            with st.spinner("Classifying five recordings..."):
                rows, correct = [], 0
                for k, c in enumerate(DEMO["classes"]):
                    a = DEMO["data"][k]
                    W = [a[i:i + WLEN].T
                         for i in range(0, len(a) - WLEN + 1, HOP)]
                    X = (pd.DataFrame([features_for_window(w) for w in W])
                         .reindex(columns=FEATURE_COLS))
                    Xv = np.nan_to_num(X.to_numpy(), nan=0.0, posinf=0.0,
                                       neginf=0.0)
                    p = MODEL["classifier"].predict_proba(Xv).mean(0)
                    h = float(np.median(MODEL["health_cov"].mahalanobis(
                        MODEL["health_scaler"].transform(Xv))))
                    got = CLASSES[int(p.argmax())]
                    correct += got == c
                    rows.append({"recording": c, "diagnosed as": got,
                                 "confidence": f"{p.max():.1%}",
                                 "health index": f"{h:,.0f}",
                                 "alarm": "yes" if h > THRESH else "no",
                                 "result": "pass" if got == c else "FAIL"})
                st.session_state["selfcheck"] = (rows, correct, len(W))

    if os.path.exists(STRICT_TABLE):
        st.divider()
        st.caption("Stricter, and only available where the EDA outputs are: "
                   "compares recomputed feature values against the exact ones "
                   "the model was trained on.")
        if st.button("Compare against the EDA table", key="sc_strict"):
            try:
                Ftab = pd.read_parquet(STRICT_TABLE)
                sample = Ftab.sample(3, random_state=0)
                worst = 0.0
                for _, row in sample.iterrows():
                    rec = pd.read_parquet(
                        os.path.join(PARQ, f"1500RPM_{row['class']}.parquet"))
                    rec = rec.iloc[:, :4]
                    rec.columns = ["t"] + AXES
                    rec = rec.dropna()
                    i = int(round(row["t_start"] * FS))
                    w = rec[AXES].to_numpy()[i:i + WLEN].T
                    got = features_for_window(w)
                    for c in FEATURE_COLS:
                        a, b = got.get(c, np.nan), row[c]
                        if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-9:
                            worst = max(worst, abs(a - b) / abs(b))
                if worst < 1e-6:
                    st.success(f"Match. Worst relative difference {worst:.2e} "
                               f"across {len(FEATURE_COLS)} features.")
                else:
                    st.error(f"MISMATCH. Worst relative difference "
                             f"{worst:.3e}. The feature code here has drifted "
                             f"from motor_eda.py. Do not trust predictions.")
            except (FileNotFoundError, OSError) as e:
                st.info(f"Needs the full recordings too. Not found: {e}")

    if DEMO is None and not os.path.exists(STRICT_TABLE):
        st.info("Needs either the bundled example recordings or the EDA "
                "feature table. Neither is present, so there is nothing to "
                "check against.")


# ---------------------------------------------------------------------------
# Load whatever the user chose
# ---------------------------------------------------------------------------
df, dropped, source = None, 0, None
if up is not None:
    try:
        df, dropped = read_csv_any(io.BytesIO(up.getvalue()))
        source = up.name
    except (ValueError, pd.errors.ParserError) as e:
        st.error(f"Could not read that file: {e}")
        st.stop()
elif example:
    # Bundled excerpt first, so the common case needs nothing but the model
    # directory. PARQUET_DIR is the fallback for a machine that happens to have
    # the full recordings, which is a development convenience, not the demo path.
    if DEMO is not None and example in DEMO["classes"]:
        k = DEMO["classes"].index(example)
        df = demo_frame(k)
        source = (f"{example}, bundled {len(df) / DEMO['fs']:.0f} s excerpt "
                  f"from t={DEMO['start_sec'][k]:.0f} s of the recording")
    else:
        try:
            d = pd.read_parquet(os.path.join(PARQ,
                                             f"1500RPM_{example}.parquet"))
            d = d.iloc[:, :4]
            d.columns = ["t"] + AXES
            df = d.dropna().reset_index(drop=True).iloc[:WLEN * 6]
            source = f"{example} (full recording, first {len(df):,} rows)"
        except (FileNotFoundError, OSError):
            st.warning(
                f"No bundled examples (`{ART}/45_demo_recordings.npz`) and no "
                f"recordings in `{os.path.abspath(PARQ)}`. Run "
                f"`export_demo_data.py` on Kaggle to create the bundle, or "
                f"upload a CSV.")

st.title("Bearing fault diagnosis")

# Rendered here rather than in the sidebar so it is legible on a projector, and
# above the analysis so it is the first thing seen after being run.
if "selfcheck" in st.session_state:
    rows, correct, n_win = st.session_state["selfcheck"]
    n = len(rows)
    st.subheader("Self-check")
    st.caption(f"All five known recordings put through the full pipeline, "
               f"{n_win} windows each. Each row should be diagnosed as itself.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if correct == n:
        st.success(f"**{correct} of {n} correct.** Windowing, feature "
                   f"extraction, scaling, the classifier and the health index "
                   f"all agree with the model as it was trained. Healthy is "
                   f"the only recording that does not raise the alarm.")
    else:
        st.error(f"**Only {correct} of {n} correct.** Something in the chain has "
                 f"drifted from the trained model. Predictions on this page "
                 f"should not be trusted until this is resolved.")
    if st.button("Hide self-check"):
        del st.session_state["selfcheck"]
        st.rerun()
    st.divider()

if df is None:
    st.info("Upload a CSV on the left, or pick a known recording to see how "
            "the page works.")
    st.subheader("What each fault looks like")
    st.caption("Envelope spectra of the five known conditions, each normalised "
               "by its own noise floor. The coloured lines are the bearing's "
               "defect frequencies. A fault puts energy on its own line.")
    fig, axs = plt.subplots(len(CLASSES), 1, figsize=(11, 1.5 * len(CLASSES)),
                            sharex=True, sharey=True)
    for ax, c, curve in zip(axs, REF["classes"], REF["env_ref"]):
        prom = curve / (np.median(curve) + 1e-20)
        ax.plot(REF["env_freq"], prom, lw=0.8,
                color=CLASS_COLOR.get(str(c), MUTED))
        for nm, hz in DEFECT_HZ.items():
            for h in (1, 2, 3):
                if h * hz <= FMAX_ENV:
                    ax.axvline(h * hz, color=DEFECT_COLOR[nm], lw=0.7,
                               alpha=0.8 if h == 1 else 0.3)
        ax.text(0.005, 0.85, str(c), transform=ax.transAxes, fontsize=8,
                va="top")
        ax.set_ylabel("x floor")
    axs[-1].set_xlabel("envelope frequency (Hz)")
    axs[0].legend(handles=[mpl.lines.Line2D([], [], color=DEFECT_COLOR[n],
                                            label=f"{n} {DEFECT_HZ[n]:.1f} Hz")
                           for n in ["FTF", "BSF", "BPFO", "BPFI"]],
                  ncol=4, fontsize=7, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    st.stop()

# ---------------------------------------------------------------------------
# Validate length, then run
# ---------------------------------------------------------------------------
if dropped:
    st.caption(f"Dropped {dropped} non-numeric row(s), most likely a units row "
               f"under the header.")
if len(df) < WLEN:
    st.error(f"**{len(df):,} rows is not enough.** Need at least {WLEN:,} "
             f"({CFG['window_sec']:.0f} s at {FS:.0f} Hz). The outer race "
             f"signature is at {DEFECT_HZ['BPFO']:.1f} Hz and the cage at "
             f"{DEFECT_HZ['FTF']:.1f} Hz; resolving those apart needs a window, "
             f"not a reading.")
    st.stop()

if "t" in df.columns and len(df) > 1:
    dt = np.median(np.diff(df["t"].to_numpy()))
    fs_seen = 1.0 / dt if dt > 0 else np.nan
    if np.isfinite(fs_seen) and abs(fs_seen - FS) / FS > 0.01:
        st.warning(f"The time column implies {fs_seen:.0f} Hz but the model "
                   f"expects {FS:.0f} Hz. Every frequency below will be wrong "
                   f"by that ratio.")

W, t_off = to_windows(df)
with st.spinner(f"Analysing {len(W)} window(s)..."):
    Xdf, Xv, proba, health = predict(W)

mean_p = proba.mean(0)
verdict = CLASSES[int(mean_p.argmax())]
per_win = [CLASSES[i] for i in proba.argmax(1)]
agree = float(np.mean([p == verdict for p in per_win]))
med_health = float(np.median(health))

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
st.caption(f"Source: {source} · {len(df):,} rows · {len(df) / FS:.1f} s · "
           f"{len(W)} window(s)")

if med_health > OOD:
    st.error(f"**Out of distribution.** Health score {med_health:,.0f} is "
             f"beyond anything seen in training (max {OOD:,.0f}). This "
             f"recording does not resemble the rig the model was built on. "
             f"Something is wrong, but the class label below should not be "
             f"trusted.")

# The health index is the primary alarm and the classifier is the secondary
# probable-cause hint, so a disagreement between them must never be rendered as
# a reassuring green box. The classifier has to pick one of five known classes
# whatever it is shown; on a recording unlike anything in training it will pick
# the nearest and report high confidence, because confidence is relative to
# those five and nothing else. Only the health index can say "none of the above".
disagree = verdict == "Healthy" and med_health > THRESH
# The mirror case. The classifier names a fault while the detector sees nothing
# unusual, which cannot both be true: every fault window in training scored above
# this threshold, so a genuine fault the model recognises should raise the alarm
# too. Showing a confident red fault panel here would overstate what is known.
unconfirmed = verdict != "Healthy" and med_health <= THRESH

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    if verdict == "Healthy" and not disagree:
        st.success(f"### {verdict}")
    elif disagree:
        st.warning(f"### {verdict}, but flagged anomalous")
    elif unconfirmed:
        st.warning(f"### {verdict}, unconfirmed")
    else:
        st.error(f"### {verdict}")
    st.caption(f"{agree:.0%} of windows agree")

if unconfirmed:
    st.warning(
        f"**The two models disagree.** The classifier names {verdict}, but the "
        f"health index reads {med_health:,.0f}, inside the healthy range "
        f"(alarm at {THRESH:,.0f}). In training every fault window scored above "
        f"that threshold, so a real fault of a known type should have tripped "
        f"it. Treat this as **inconclusive**: the classifier is obliged to "
        f"return one of five labels even when the honest answer is that this "
        f"does not look like any of them.")

if disagree:
    st.warning(
        f"**The two models disagree, and the alarm wins.** The classifier says "
        f"Healthy, but the health index reads {med_health:,.0f} against an alarm "
        f"threshold of {THRESH:,.0f}, which is {med_health / THRESH:.0f} times "
        f"over. The classifier can only choose between the five conditions it "
        f"was trained on, so on an unfamiliar recording it picks the closest one "
        f"and still reports high confidence. The health index is trained on "
        f"healthy data alone and is the one that can say none of the above. "
        f"Treat this as **anomalous, cause unknown**, not as healthy.")
with c2:
    st.metric("Confidence", f"{mean_p.max():.1%}")
with c3:
    st.metric("Health index", f"{med_health:,.0f}",
              delta=f"alarm above {THRESH:,.0f}",
              delta_color="inverse" if med_health > THRESH else "normal")

# The sentence that makes this explainable rather than a black box.
env_feats = [c for c in FEATURE_COLS if "_env_" in c and "_h1" in c]
if env_feats:
    ref_v = FEATREF[FEATREF["class"] == verdict].set_index("feature")["median"]
    ref_h = FEATREF[FEATREF["class"] == "Healthy"].set_index("feature")["median"]
    got = Xdf[env_feats].median()
    lead = max(env_feats, key=lambda c: got[c] / (ref_h.get(c, 1) + 1e-9)
               if np.isfinite(got[c]) else 0)
    nm = lead.split("_env_")[1].split("_h")[0]
    st.info(f"**Why:** energy at **{nm}**, the "
            f"{ {'BPFO': 'outer race', 'BPFI': 'inner race', 'BSF': 'ball', 'FTF': 'cage'}[nm] } "
            f"frequency ({DEFECT_HZ[nm]:.1f} Hz), is **{got[lead]:.1f}x** the "
            f"local noise floor in this recording. Healthy is typically "
            f"{ref_h.get(lead, float('nan')):.1f}x, {verdict} typically "
            f"{ref_v.get(lead, float('nan')):.1f}x.")

# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Envelope spectrum", "Compare with known faults", "Raw signal and health",
     "Where it sits"])

ax_i = AXES.index(CFG["envelope_axis"])

with tab1:
    st.caption("This is the diagnosis. The envelope spectrum reveals the "
               "repetition rate of the bearing impacts. Each coloured line is "
               "one of the bearing's defect frequencies; a fault puts a peak "
               "and its harmonics on its own line.")
    fe, Pe = envelope_spectrum(W[len(W) // 2][ax_i].astype(np.float64), FS,
                               BAND_LO, BAND_HI, WLEN)
    m = fe <= FMAX_ENV
    amp = np.sqrt(Pe[m]) / (np.median(np.sqrt(Pe[m])) + 1e-20)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(fe[m], amp, lw=0.9, color=CLASS_COLOR.get(verdict, FG))
    ax.axhline(4, color=MUTED, ls=(0, (4, 3)), lw=0.8,
               label="4x floor, the line above which a peak is convincing")
    for nm_, hz in DEFECT_HZ.items():
        for h in (1, 2, 3):
            if h * hz <= FMAX_ENV:
                ax.axvline(h * hz, color=DEFECT_COLOR[nm_], lw=0.8,
                           alpha=0.85 if h == 1 else 0.3)
    ax.set_xlabel(f"envelope frequency (Hz), axis {CFG['envelope_axis']}")
    ax.set_ylabel("amplitude / local noise floor")
    ax.legend(handles=[mpl.lines.Line2D([], [], color=DEFECT_COLOR[n],
                                        label=f"{n} {DEFECT_HZ[n]:.1f} Hz")
                       for n in ["FTF", "BSF", "BPFO", "BPFI"]]
                      + [mpl.lines.Line2D([], [], color=MUTED, ls=(0, (4, 3)),
                                          label="4x floor")],
              ncol=5, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("Measured height at each defect frequency, in multiples of the "
               "local noise floor.")
    snr = []
    for nm_, hz in DEFECT_HZ.items():
        for h in (1, 2, 3):
            v = peak_over_floor(fe, Pe, h * hz)[1]
            snr.append({"line": f"{nm_} x{h}", "Hz": round(h * hz, 2),
                        "x floor": round(v, 2) if np.isfinite(v) else None,
                        "convincing": "yes" if np.isfinite(v) and v > 4 else ""})
    st.dataframe(pd.DataFrame(snr), hide_index=True, width="stretch")

with tab2:
    st.caption("Your recording against the five known conditions, all on the "
               "same scale.")
    fig, axs = plt.subplots(len(CLASSES) + 1, 1,
                            figsize=(11, 1.4 * (len(CLASSES) + 1)),
                            sharex=True, sharey=True)
    axs[0].plot(fe[m], amp, lw=0.9, color=FG)
    axs[0].text(0.005, 0.85, "YOUR RECORDING", transform=axs[0].transAxes,
                fontsize=8, va="top", weight="bold")
    for ax, c, curve in zip(axs[1:], REF["classes"], REF["env_ref"]):
        ax.plot(REF["env_freq"], curve / (np.median(curve) + 1e-20), lw=0.8,
                color=CLASS_COLOR.get(str(c), MUTED))
        ax.text(0.005, 0.85, str(c), transform=ax.transAxes, fontsize=8,
                va="top")
    for ax in axs:
        for nm_, hz in DEFECT_HZ.items():
            for h in (1, 2, 3):
                if h * hz <= FMAX_ENV:
                    ax.axvline(h * hz, color=DEFECT_COLOR[nm_], lw=0.7,
                               alpha=0.8 if h == 1 else 0.25)
        ax.set_ylabel("x floor")
    axs[-1].set_xlabel("envelope frequency (Hz)")
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    cA, cB = st.columns(2)
    with cA:
        st.caption("Raw vibration, first 200 ms, against a healthy reference "
                   "on the same scale.")
        n = int(0.2 * FS)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        hi_ = list(REF["classes"]).index("Healthy")
        ax.plot(np.arange(n) / FS * 1000, REF["raw_ref"][hi_][:n], lw=0.6,
                color=CLASS_COLOR["Healthy"], label="Healthy reference")
        ax.plot(np.arange(n) / FS * 1000, W[len(W) // 2][ax_i][:n], lw=0.6,
                color=FG, label="yours")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("acceleration (g)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
    with cB:
        st.caption("Health index. Trained on healthy data only, so it flags "
                   "anything unusual including fault types never recorded.")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for c in REF["classes"]:
            s = REF["health_scores"][REF["health_labels"] == c]
            ax.hist(s[np.isfinite(s)], bins=40, histtype="step", lw=1.1,
                    density=True, color=CLASS_COLOR.get(str(c), MUTED),
                    label=str(c))
        ax.axvline(med_health, color=FG, lw=2.0, label="yours")
        ax.axvline(THRESH, color=MUTED, ls=(0, (4, 3)), lw=1.0, label="alarm")
        ax.set_xscale("log")
        ax.set_xlabel("health index")
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        st.pyplot(fig)

    if len(W) > 1:
        st.caption("Per window over the recording. Steady means the condition "
                   "is stable; a trend would be the beginning of a "
                   "degradation curve.")
        fig, ax = plt.subplots(figsize=(11, 2.6))
        ax.plot(t_off, health, marker="o", ms=3, color=FG)
        ax.axhline(THRESH, color=MUTED, ls=(0, (4, 3)), lw=1.0, label="alarm")
        ax.set_yscale("log")
        ax.set_xlabel("time into recording (s)")
        ax.set_ylabel("health index")
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    cA, cB = st.columns([1, 1])
    with cA:
        st.caption("Class probabilities, averaged over every window.")
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.barh(CLASSES, mean_p,
                color=[CLASS_COLOR.get(c, MUTED) for c in CLASSES])
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        plt.tight_layout()
        st.pyplot(fig)
    with cB:
        st.caption("The 64 features reduced to two dimensions. Your windows in "
                   "black, against the known recordings.")
        Z = (Xv - REF["pca_mean"]) / REF["pca_scale"]
        P2 = Z @ REF["pca_components"].T
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for c in REF["classes"]:
            m2 = REF["pca_labels"] == c
            ax.scatter(REF["pca_points"][m2, 0], REF["pca_points"][m2, 1], s=8,
                       alpha=0.55, edgecolor="none",
                       color=CLASS_COLOR.get(str(c), MUTED), label=str(c))
        ax.scatter(P2[:, 0], P2[:, 1], s=70, marker="X", color=FG,
                   edgecolor=BG, linewidth=0.8, label="yours", zorder=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=6)
        plt.tight_layout()
        st.pyplot(fig)

    st.caption("Per window detail.")
    st.dataframe(pd.DataFrame({
        "window": np.arange(len(W)),
        "starts at (s)": np.round(t_off, 2),
        "prediction": per_win,
        "confidence": np.round(proba.max(1), 3),
        "health index": np.round(health, 1),
        "alarm": np.where(health > THRESH, "yes", ""),
    }), hide_index=True, width="stretch")

st.divider()
st.caption(CFG["scope_limits"])
