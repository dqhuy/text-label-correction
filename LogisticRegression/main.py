import streamlit as st
import pandas as pd
from db_ops import load_replay_data, save_corrected_data
from train_ops import train_field_models, predict_with_field_models, check_and_retrain_field_models
from typing import List, Dict
import os

# Khởi tạo session state
if 'documents' not in st.session_state:
    st.session_state['documents'] = []
if 'current_doc_index' not in st.session_state:
    st.session_state['current_doc_index'] = 0
if 'corrected_docs' not in st.session_state:
    st.session_state['corrected_docs'] = []
if 'field_models' not in st.session_state:
    st.session_state['field_models'] = {}
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = []
if 'existing_labels' not in st.session_state:
    st.session_state['existing_labels'] = set()
if 'validation_results' not in st.session_state:
    st.session_state['validation_results'] = []
if 'auto_validate' not in st.session_state:
    st.session_state['auto_validate'] = False
if 'batch_count' not in st.session_state:
    st.session_state['batch_count'] = 0

def save_validation_results(results: List[Dict]):
    """Lưu kết quả validation vào CSV."""
    df = pd.DataFrame(results)
    output_path = "replay/validation_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def compute_field_accuracy(results: List[Dict]) -> Dict[str, float]:
    """Tính độ chính xác trung bình theo doctypefieldcode."""
    field_counts = {}
    field_correct = {}
    for result in results:
        field_code = result['doctypefieldcode']
        is_correct = result['predict_value'] == result['human_value']
        field_counts[field_code] = field_counts.get(field_code, 0) + 1
        field_correct[field_code] = field_correct.get(field_code, 0) + (1 if is_correct else 0)
    
    accuracy = {
        field: correct / count if count > 0 else 0.0
        for field, correct, count in [
            (field, field_correct.get(field, 0), field_counts.get(field, 1))
            for field in field_counts
        ]
    }
    return accuracy

def main():
    st.title("OCR Correction Replay")

    # Load dữ liệu
    if not st.session_state.documents:
        try:
            st.session_state.documents = load_replay_data()
            # Sắp xếp tài liệu theo docid
            st.session_state.documents.sort(key=lambda x: x['docid'])
            st.success("Dữ liệu đã được tải!")
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {e}")
            return

    total_docs = len(st.session_state.documents)

    # Chế độ validate tự động
    st.sidebar.header("Chế độ xử lý")
    if st.sidebar.button("Bật chế độ validate tự động"):
        st.session_state.auto_validate = True
        st.session_state.current_doc_index = 0
        st.session_state.validation_results = []
        st.session_state.batch_count = 0
        st.rerun()

    if st.sidebar.button("Tắt chế độ validate tự động"):
        st.session_state.auto_validate = False
        st.rerun()

    # Hiển thị độ chính xác trung bình
    if st.session_state.validation_results:
        st.subheader("Độ chính xác trung bình theo trường")
        accuracy = compute_field_accuracy(st.session_state.validation_results)
        accuracy_df = pd.DataFrame([
            {"Mã trường": field, "Độ chính xác (%)": acc * 100}
            for field, acc in accuracy.items()
        ])
        st.table(accuracy_df)

    # Kiểm tra nếu đã hoàn tất
    if st.session_state.current_doc_index >= total_docs:
        st.success("Hoàn tất nhập liệu!")
        if st.button("Lưu dữ liệu đã sửa"):
            try:
                save_corrected_data(st.session_state.corrected_docs)
                st.success("Dữ liệu đã được lưu vào replay/corrected_data.csv")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")
        if st.session_state.validation_results:
            try:
                save_validation_results(st.session_state.validation_results)
                st.success("Kết quả validation đã được lưu vào replay/validation_results.csv")
            except Exception as e:
                st.error(f"Lỗi khi lưu kết quả validation: {e}")
        return

    # Hiển thị tài liệu hiện tại
    current_doc = st.session_state.documents[st.session_state.current_doc_index]
    st.header(f"Tài liệu: {current_doc['docid']}")
    st.write(f"Tài liệu {st.session_state.current_doc_index + 1}/{total_docs}")

    # Tạo bảng dữ liệu
    table_data = []
    for idx, field in enumerate(current_doc['fields']):
        row = {
            "Mã trường": field['doctypefieldcode'],
            "OCR value": field['ocr_value'],
            "Human value": field['human_value'],
            "Predict value": "(Chưa có dự đoán)",
            "Confident (%)": "-",
            "_match": False
        }
        if st.session_state.current_doc_index >= 10 and st.session_state.predictions and idx < len(st.session_state.predictions):
            predict_value, confidence = st.session_state.predictions[idx]
            row["Predict value"] = predict_value
            row["Confident (%)"] = f"{confidence*100:.2f}"
            row["_match"] = predict_value == field['human_value']
        table_data.append(row)

    # Hiển thị bảng với màu
    st.subheader("Thông tin chi tiết")
    df = pd.DataFrame(table_data)
    def highlight_row(row):
        color = 'background-color: lightgreen' if row['_match'] else ''
        return [color] * len(row)
    styled_df = df.style.apply(highlight_row, axis=1).hide(['_match'], axis='columns')
    st.dataframe(styled_df, use_container_width=True)

    if not st.session_state.auto_validate:
        # Chế độ thủ công
        if st.button("OK"):
            # Lưu dữ liệu
            corrected_doc = {
                'docid': current_doc['docid'],
                'fields': [
                    {
                        **field,
                        'corrected_human_value': field['human_value']
                    }
                    for field in current_doc['fields']
                ]
            }
            st.session_state.corrected_docs.append(corrected_doc)

            # Huấn luyện
            if st.session_state.current_doc_index >= 9 and len(st.session_state.corrected_docs) >= 10:
                try:
                    new_models = train_field_models(st.session_state.corrected_docs[-10:])
                    st.session_state.field_models.update(new_models)
                    st.session_state.existing_labels = set().union(
                        *[model.labels for model in st.session_state.field_models.values()]
                    )
                    st.success("Các model đã được huấn luyện!")
                except Exception as e:
                    st.error(f"Lỗi khi huấn luyện model: {e}")

            # Dự đoán
            if st.session_state.field_models and st.session_state.current_doc_index + 1 < total_docs:
                next_doc = st.session_state.documents[st.session_state.current_doc_index + 1]
                try:
                    st.session_state.predictions = predict_with_field_models(
                        next_doc['fields'], st.session_state.field_models
                    )
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")

            # Huấn luyện bổ sung
            if st.session_state.current_doc_index >= 9 and len(st.session_state.corrected_docs) >= 10:
                try:
                    check_and_retrain_field_models(
                        st.session_state.field_models,
                        st.session_state.corrected_docs[-10:],
                        st.session_state.existing_labels
                    )
                    st.session_state.existing_labels = set().union(
                        *[model.labels for model in st.session_state.field_models.values()]
                    )
                    st.success("Các model đã được huấn luyện bổ sung với nhãn mới!")
                except Exception as e:
                    st.error(f"Lỗi huấn luyện bổ sung: {e}")

            st.session_state.current_doc_index += 1
            st.rerun()
    else:
        # Chế độ validate tự động
        batch_size = 10
        start_idx = st.session_state.batch_count * batch_size
        end_idx = min(start_idx + batch_size, total_docs)

        if start_idx >= total_docs:
            st.session_state.auto_validate = False
            st.rerun()

        # Xử lý batch
        for idx in range(start_idx, end_idx):
            if idx >= total_docs:
                break
            current_doc = st.session_state.documents[idx]
            corrected_doc = {
                'docid': current_doc['docid'],
                'fields': [
                    {
                        **field,
                        'corrected_human_value': field['human_value']
                    }
                    for field in current_doc['fields']
                ]
            }
            st.session_state.corrected_docs.append(corrected_doc)

            # Lưu kết quả validation
            if idx >= 10 and st.session_state.predictions:
                for field_idx, field in enumerate(current_doc['fields']):
                    if field_idx < len(st.session_state.predictions):
                        predict_value, confidence = st.session_state.predictions[field_idx]
                        st.session_state.validation_results.append({
                            'docid': current_doc['docid'],
                            'doctypefieldcode': field['doctypefieldcode'],
                            'ocr_value': field['ocr_value'],
                            'human_value': field['human_value'],
                            'predict_value': predict_value,
                            'confidence': confidence
                        })

        # Huấn luyện sau mỗi 10 tài liệu
        if end_idx % batch_size == 0 and len(st.session_state.corrected_docs) >= 10:
            try:
                new_models = train_field_models(st.session_state.corrected_docs[-batch_size:])
                st.session_state.field_models.update(new_models)
                st.session_state.existing_labels = set().union(
                    *[model.labels for model in st.session_state.field_models.values()]
                )
                st.success(f"Đã huấn luyện model cho batch {st.session_state.batch_count + 1}")
            except Exception as e:
                st.error(f"Lỗi khi huấn luyện model: {e}")

        # Dự đoán cho batch tiếp theo
        if st.session_state.field_models and end_idx < total_docs:
            next_doc = st.session_state.documents[end_idx]
            try:
                st.session_state.predictions = predict_with_field_models(
                    next_doc['fields'], st.session_state.field_models
                )
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

        # Huấn luyện bổ sung
        if end_idx % batch_size == 0 and len(st.session_state.corrected_docs) >= 10:
            try:
                check_and_retrain_field_models(
                    st.session_state.field_models,
                    st.session_state.corrected_docs[-batch_size:],
                    st.session_state.existing_labels
                )
                st.session_state.existing_labels = set().union(
                    *[model.labels for model in st.session_state.field_models.values()]
                )
                st.success(f"Đã huấn luyện bổ sung model cho batch {st.session_state.batch_count + 1}")
            except Exception as e:
                st.error(f"Lỗi huấn luyện bổ sung: {e}")

        st.session_state.batch_count += 1
        st.session_state.current_doc_index = end_idx
        st.rerun()

if __name__ == "__main__":
    main()