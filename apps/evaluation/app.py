# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from modules.api import LlamaStackEvaluation

from modules.utils import process_dataset

EVALUATION_API = LlamaStackEvaluation()


def main():
    # Add collapsible sidebar
    with st.sidebar:
        # Add collapse button
        if "sidebar_state" not in st.session_state:
            st.session_state.sidebar_state = True

        if st.session_state.sidebar_state:
            st.title("Navigation")
            page = st.radio(
                "Select a Page",
                ["Dataset Upload", "Analysis", "Settings"],
                index=0,
            )
        else:
            page = "Dataset Upload"  # Default page when sidebar is collapsed

    # Main content area
    st.title("🦙 Llama Stack Evaluations")

    if page == "Dataset Upload":
        dataset_upload_page()
    elif page == "Analysis":
        analysis_page()
    elif page == "Settings":
        settings_page()


def dataset_upload_page():
    # File uploader
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.error("No file uploaded")
        return

    # Process uploaded file
    df = process_dataset(uploaded_file)
    if df is None:
        st.error("Error processing file")
        return

    # Display dataset information
    st.success("Dataset loaded successfully!")

    # Display dataframe preview
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Select Scoring Functions to Run Evaluation On
    st.subheader("Select Scoring Functions")
    scoring_functions = EVALUATION_API.list_scoring_functions()
    scoring_functions_descriptions = {
        sf.identifier: sf.description for sf in scoring_functions
    }
    scoring_functions_names = list(scoring_functions_descriptions.keys())
    selected_scoring_functions = st.multiselect(
        "Select 1 or more scoring functions to run evaluation",
        options=scoring_functions_names,
        help="Choose one or more scoring functions.",
    )

    if selected_scoring_functions:
        st.write("Selected:")
        for scoring_function in selected_scoring_functions:
            st.write(
                f"- **{scoring_function}**: {scoring_functions_descriptions[scoring_function]}"
            )

        # Add run evaluation button
        if st.button("Run Evaluation"):
            st.write("Running evaluation...")


def analysis_page():
    st.header("Analysis")
    st.write("Analysis page content goes here")


def settings_page():
    st.header("Settings")
    st.write("Settings page content goes here")


if __name__ == "__main__":
    main()
