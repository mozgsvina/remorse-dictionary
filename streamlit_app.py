import streamlit as st

import streamlit as st
import pandas as pd

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_PATH = "data/сводный_словарь.xlsx"

st.set_page_config(
    page_title="rEmoRSe Lexicon Explorer",
    layout="wide",
)


# -------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_lexicon(path_or_url: str) -> pd.DataFrame:
    df = pd.read_excel(path_or_url)

    # Normalize column names (just in case)
    df.columns = [c.strip() for c in df.columns]

    # Ensure core columns exist
    expected_cols = [
        "lemma",
        "emo_tag",
        "rss_auto",
        "rss_markup",
        "RuSentiLex",
        "LINIS_Crowd",
        "NRC Word-Emotion Association Lexicon",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns: {missing}")

    # Make rss_auto / rss_markup numeric 0/1 (or counts; treat >0 as present)
    for col in ["rss_auto", "rss_markup"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Helper booleans for external lexicons
    df["in_rusentilex"] = df["RuSentiLex"].notna()
    df["in_linis"] = df["LINIS_Crowd"].notna()
    df["in_nrc"] = df["NRC Word-Emotion Association Lexicon"].notna()

    # Source type: auto / manual / both / none
    def source_type(row):
        a = row.get("rss_auto", 0) > 0
        m = row.get("rss_markup", 0) > 0
        if a and m:
            return "auto + manual"
        elif a:
            return "auto only"
        elif m:
            return "manual only"
        else:
            return "none"

    df["source_type"] = df.apply(source_type, axis=1)

    return df


df = load_lexicon(DATA_PATH)


# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------

st.sidebar.title("Filters")

# Lemma search (substring)
lemma_query = st.sidebar.text_input(
    "Search lemma (substring, case-insensitive)", value=""
)

# emo_tag multiselect
all_emo_tags = sorted(df["emo_tag"].dropna().unique())
selected_emo_tags = st.sidebar.multiselect(
    "Emotion tags (emo_tag)",
    options=all_emo_tags,
    default=[],
)

# Source type
source_options = ["auto only", "manual only", "auto + manual"]
selected_sources = st.sidebar.multiselect(
    "Source type",
    options=source_options,
    default=[],
)

# External lexicons
st.sidebar.markdown("### External lexicons")
only_in_rusentilex = st.sidebar.checkbox("Only lemmas present in RuSentiLex")
only_in_linis = st.sidebar.checkbox("Only lemmas present in LINIS Crowd")
only_in_nrc = st.sidebar.checkbox("Only lemmas present in NRC")

st.sidebar.markdown("---")
st.sidebar.caption("rEmoRSe demo explorer")


# -------------------------------------------------------------------
# MAIN LAYOUT: TABS
# -------------------------------------------------------------------

st.title("rEmoRSe: Russian Emotional Lexicon Explorer")

st.markdown(
    """
This demo app lets you explore the **combined emotional lexicon** rEmoRSe,  
built from:

- automatic identification of emotional words in a corpus (fastText-based),
- manual lexical annotation of Russian short stories,
- and overlaps with other sentiment/emotion lexicons (RuSentiLex, LINIS Crowd, NRC).
"""
)

tab_explorer, tab_stats, tab_coverage = st.tabs(
    ["🔍 Lexicon explorer", "📊 Emotion statistics", "🌐 Cross-lexicon coverage"]
)


# -------------------------------------------------------------------
# COMMON FILTERING FUNCTION
# -------------------------------------------------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    # Lemma search
    if lemma_query:
        q = lemma_query.lower()
        filtered = filtered[filtered["lemma"].str.lower().str.contains(q)]

    # emo_tag filter
    if selected_emo_tags:
        filtered = filtered[filtered["emo_tag"].isin(selected_emo_tags)]

    # source_type filter
    if selected_sources:
        filtered = filtered[filtered["source_type"].isin(selected_sources)]

    # external lexicons
    if only_in_rusentilex:
        filtered = filtered[filtered["in_rusentilex"]]
    if only_in_linis:
        filtered = filtered[filtered["in_linis"]]
    if only_in_nrc:
        filtered = filtered[filtered["in_nrc"]]

    return filtered


# -------------------------------------------------------------------
# TAB 1: LEXICON EXPLORER
# -------------------------------------------------------------------

with tab_explorer:
    st.subheader("Lexicon explorer")

    filtered = apply_filters(df)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total lemma–emotion pairs", value=f"{len(df):,}")
    with col_b:
        st.metric("Pairs after filters", value=f"{len(filtered):,}")
    with col_c:
        st.metric("Unique lemmas (filtered)", value=f"{filtered['lemma'].nunique():,}")

    st.markdown("### Filtered entries")

    # Reorder columns for nicer display
    display_cols = [
        "lemma",
        "emo_tag",
        "source_type",
        "rss_auto",
        "rss_markup",
        "RuSentiLex",
        "LINIS_Crowd",
        "NRC Word-Emotion Association Lexicon",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values(["lemma", "emo_tag"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Lemma details")
    if not filtered.empty:
        lemma_choices = sorted(filtered["lemma"].unique())
        selected_lemma = st.selectbox(
            "Pick a lemma to inspect", options=lemma_choices
        )

        lemma_rows = df[df["lemma"] == selected_lemma]

        st.write(f"All emotion entries for **{selected_lemma}**:")
        st.dataframe(
            lemma_rows[display_cols].sort_values("emo_tag"),
            use_container_width=True,
            hide_index=True,
        )

        # Quick summary: which lexicons contain this lemma?
        in_sets = []
        if lemma_rows["in_rusentilex"].any():
            in_sets.append("RuSentiLex")
        if lemma_rows["in_linis"].any():
            in_sets.append("LINIS Crowd")
        if lemma_rows["in_nrc"].any():
            in_sets.append("NRC")

        st.markdown(
            f"- Present in external lexicons: "
            + (", ".join(in_sets) if in_sets else "_none of the selected ones_")
        )

        # Download subset
        csv = lemma_rows.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download lemma rows as CSV",
            data=csv,
            file_name=f"rEmoRSe_{selected_lemma}.csv",
            mime="text/csv",
        )
    else:
        st.info("No rows match the current filters.")


# -------------------------------------------------------------------
# TAB 2: EMOTION STATISTICS
# -------------------------------------------------------------------

with tab_stats:
    st.subheader("Emotion statistics")

    filtered = apply_filters(df)

    st.markdown("#### Distribution of emo_tag (after filters)")
    emo_counts = (
        filtered["emo_tag"]
        .value_counts()
        .rename_axis("emo_tag")
        .reset_index(name="count")
        .sort_values("emo_tag")
    )

    if emo_counts.empty:
        st.info("No data for current filters.")
    else:
        st.dataframe(emo_counts, use_container_width=True, hide_index=True)
        st.bar_chart(
            emo_counts.set_index("emo_tag")["count"]
        )

    st.markdown("#### Source-type breakdown by emotion")
    if not filtered.empty:
        source_pivot = (
            filtered.groupby(["emo_tag", "source_type"])
            .size()
            .reset_index(name="count")
        )
        st.dataframe(source_pivot, use_container_width=True, hide_index=True)

        # Optionally, a pivot table view
        pivot_table = source_pivot.pivot(
            index="emo_tag", columns="source_type", values="count"
        ).fillna(0).astype(int)
        st.markdown("Pivot view (rows = emo_tag, columns = source_type):")
        st.dataframe(pivot_table, use_container_width=True)

    st.markdown("#### Global summary (ignoring filters)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Unique lemmas", f"{df['lemma'].nunique():,}")
    with col2:
        st.metric("Total lemma–emotion pairs", f"{len(df):,}")
    with col3:
        st.metric("Pairs with auto annotation", f"{(df['rss_auto'] > 0).sum():,}")
    with col4:
        st.metric("Pairs with manual markup", f"{(df['rss_markup'] > 0).sum():,}")


# -------------------------------------------------------------------
# TAB 3: CROSS-LEXICON COVERAGE
# -------------------------------------------------------------------

with tab_coverage:
    st.subheader("Cross-lexicon coverage")

    # Per-lexicon coverage (binary: is lemma present anywhere in that lexicon?)
    lex_cols = ["in_rusentilex", "in_linis", "in_nrc"]
    for col_name, label in zip(
        lex_cols, ["RuSentiLex", "LINIS Crowd", "NRC"]
    ):
        n_pairs = df[col_name].sum()
        n_lemmas = df.loc[df[col_name], "lemma"].nunique()
        st.write(
            f"- **{label}**: {n_pairs:,} lemma–emotion pairs, "
            f"{n_lemmas:,} unique lemmas (within rEmoRSe)."
        )

    st.markdown("#### Combination of external lexicons (per lemma)")

    # Build a compact "signature" per lemma like "RuSentiLex + NRC", etc.
    def lex_signature(subdf: pd.DataFrame) -> str:
        labels = []
        if subdf["in_rusentilex"].any():
            labels.append("RuSentiLex")
        if subdf["in_linis"].any():
            labels.append("LINIS")
        if subdf["in_nrc"].any():
            labels.append("NRC")
        return " + ".join(labels) if labels else "rEmoRSe only"

    lemma_signatures = (
        df.groupby("lemma")
        .apply(lex_signature)
        .reset_index(name="signature")
    )

    signature_counts = (
        lemma_signatures["signature"]
        .value_counts()
        .rename_axis("signature")
        .reset_index(name="lemma_count")
    )

    st.dataframe(signature_counts, use_container_width=True, hide_index=True)
    st.bar_chart(
        signature_counts.set_index("signature")["lemma_count"]
    )

    st.caption(
        "Counts here are per lemma (not per lemma–emotion pair). "
        "This gives a rough overview of how rEmoRSe intersects with other resources."
    )

