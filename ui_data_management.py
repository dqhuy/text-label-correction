import streamlit as st
import pandas as pd
from db_ops import get_categories, add_category, get_labels_by_category, add_label, get_abbreviations, add_abbreviation, update_abbreviation, delete_abbreviation
import logging

from train_ops import update_faiss_index

logger = logging.getLogger(__name__)

def data_management_page():
    """Render the Data Management interface."""
    st.header("Data Management: Categories, Labels, and Abbreviations")
    
    st.subheader("Categories")
    categories = get_categories()
    if categories:
        df = pd.DataFrame(categories)
        st.dataframe(df)
        category_id = st.selectbox("Select Category to View Labels",
                                   [c["id"] for c in categories],
                                   format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
        labels = get_labels_by_category(category_id)
        if labels:
            st.subheader(f"Labels in {next(c['name'] for c in categories if c['id'] == category_id)}")
            df_labels = pd.DataFrame(labels)
            df_labels["is_dynamic"] = df_labels["is_dynamic"].map({0: "Static", 1: "Dynamic"})
            st.dataframe(df_labels)
        else:
            st.write("No labels in this category.")
    else:
        st.write("No categories available.")

    st.subheader("Add New Category")
    category_name = st.text_input("Category Name")
    if st.button("Add Category"):
        if category_name:
            add_category(category_name)
            st.success(f"Added category: {category_name}")
            st.experimental_rerun()
        else:
            st.error("Please enter a category name.")

    st.subheader("Add Label to Category")
    if categories:
        category_id = st.selectbox("Select Category for Label",
                                   [c["id"] for c in categories],
                                   format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
        label_value = st.text_input("Label Value")
        is_dynamic = st.checkbox("Dynamic Label", value=True)
        if st.button("Add Label"):
            if label_value:
                add_label(label_value, category_id, 1 if is_dynamic else 0)
                st.session_state.valid_values.append({"label": label_value, "category": next(c["name"] for c in categories if c["id"] == category_id)})
                st.session_state.labels = update_faiss_index(st.session_state.model, st.session_state.index, st.session_state.valid_values)
                st.success(f"Added label: {label_value} to category")
                st.experimental_rerun()
            else:
                st.error("Please enter a label value.")
    else:
        st.error("No categories available. Add a category first.")

    st.subheader("Manage Abbreviations")
    abbrs = get_abbreviations()
    if abbrs:
        df = pd.DataFrame(abbrs)
        st.dataframe(df)
        
        st.subheader("Add Abbreviation")
        if categories:
            category_id = st.selectbox("Select Category for Abbreviation",
                                       [c["id"] for c in categories],
                                       format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
            abbr = st.text_input("Abbreviation (e.g., VN)")
            full_form = st.text_input("Full Form (e.g., Việt Nam)")
            if st.button("Add Abbreviation"):
                if abbr and full_form:
                    add_abbreviation(abbr.upper(), full_form, category_id)
                    st.success(f"Added abbreviation: {abbr} -> {full_form}")
                    st.experimental_rerun()
                else:
                    st.error("Please enter both abbreviation and full form.")
        
        st.subheader("Edit/Delete Abbreviation")
        abbr_id = st.selectbox("Select Abbreviation to Edit/Delete",
                               [a["id"] for a in abbrs],
                               format_func=lambda x: next(f"{a['abbr']} -> {a['full_form']}" for a in abbrs if a["id"] == x))
        selected_abbr = next(a for a in abbrs if a["id"] == abbr_id)
        new_abbr = st.text_input("New Abbreviation", value=selected_abbr["abbr"])
        new_full_form = st.text_input("New Full Form", value=selected_abbr["full_form"])
        new_category_id = st.selectbox("New Category",
                                       [c["id"] for c in categories],
                                       index=[c["id"] for c in categories].index(selected_abbr["category_id"]),
                                       format_func=lambda x: next(c["name"] for c in categories if c["id"] == x))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update Abbreviation"):
                if new_abbr and new_full_form:
                    update_abbreviation(abbr_id, new_abbr.upper(), new_full_form, new_category_id)
                    st.success(f"Updated abbreviation: {new_abbr} -> {new_full_form}")
                    st.experimental_rerun()
                else:
                    st.error("Please enter both abbreviation and full form.")
        with col2:
            if st.button("Delete Abbreviation"):
                delete_abbreviation(abbr_id)
                st.success(f"Deleted abbreviation: {selected_abbr['abbr']}")
                st.experimental_rerun()
    else:
        st.write("No abbreviations available.")