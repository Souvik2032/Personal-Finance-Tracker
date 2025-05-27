from flask import Flask, render_template, request, jsonify, send_file
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import pandas as pd
from fpdf import FPDF

# Explicit absolute paths for templates and static folders:
TEMPLATE_PATH = r"C:\Users\souvi\OneDrive\Desktop\Personal_Finance_Tracker\templates"
STATIC_PATH = r"C:\Users\souvi\OneDrive\Desktop\Personal_Finance_Tracker\static"

app = Flask(__name__, template_folder=TEMPLATE_PATH, static_folder=STATIC_PATH)
print("Template folder:", app.template_folder)
print("Static folder:", app.static_folder)

# Model folder path (update if needed)
model_folder = r"C:\Users\souvi\OneDrive\Desktop\BERT model\bert-transaction-model"

# Load your fine-tuned model and tokenizer.
tokenizer = BertTokenizer.from_pretrained(model_folder, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_folder, local_files_only=True)

def rule_based_prediction(message):
    msg = message.strip().lower()
    
    # Pharmacy: ~20 keywords.
    pharmacy_keywords = [
        "pharmacy", "apollo", "drugstore", "prescription", "rx",
        "meds", "medicine", "pharm", "walgreens", "rite aid", "cvs", "cv pharmacy",
        "pharmacist", "medication", "drugs", "dispensed", "pharmaceutical", 
        "pharmacies", "drug", "pharmaco"
    ]
    if any(keyword in msg for keyword in pharmacy_keywords):
        return "Pharmacy"
    
    # Credited: ~20 keywords.
    credited_keywords = [
        "credited", "salary", "deposit", "received", "refund", "bonus", "payroll", 
        "incoming", "transfer", "remittance", "payment", "advance", "compensation", 
        "inflow", "credit", "funds added", "credited amount", "income", "direct deposit", "added"
    ]
    if any(keyword in msg for keyword in credited_keywords):
        return "Credited"
    
    # Food: ~20 keywords.
    food_keywords = [
        "food", "restaurant", "dinner", "lunch", "cafe", "burger", "pizza", 
        "groceries", "eatery", "coffee", "bistro", "mcdonald", "kfc", "subway", 
        "dominos", "dine", "meal", "snack", "food court", "canteen"
    ]
    if any(keyword in msg for keyword in food_keywords):
        return "Food"
    
    # Transportation: ~20 keywords.
    transportation_keywords = [
        "metro", "taxi", "bus", "train", "cab", "fare", "uber", "ola", "lyft", 
        "ride", "commute", "ticket", "tram", "transport", "auto", "rickshaw", 
        "minibus", "shuttle", "ferry", "travel", "irctc"
    ]
    if any(keyword in msg for keyword in transportation_keywords):
        return "Transportation"
    
    # Utilities: ~20 keywords.
    utilities_keywords = [
        "bill", "electricity", "water", "internet", "phone", "gas", "cable", 
        "utility", "power", "sewage", "waste", "dth", "broadband", "electric", 
        "tariff", "maintenance", "consumption", "meter", "charge", "connection"
    ]
    if any(keyword in msg for keyword in utilities_keywords):
        return "Utilities"
    
    # Entertainment: ~20 keywords.
    entertainment_keywords = [
        "movie", "concert", "theatre", "festival", "ticket", "show", "play", 
        "cinema", "performance", "exhibition", "amusement", "cinepolis", "pvr", 
        "inox", "multiplex", "screening", "live", "gig", "drama", "comedy"
    ]
    if any(keyword in msg for keyword in entertainment_keywords):
        return "Entertainment"
    
    # Shopping: ~20 keywords.
    shopping_keywords = [
        "shopping", "purchase", "store", "mall", "boutique", "clothes", 
        "electronics", "order", "amazon", "walmart", "retail", "outlet", 
        "market", "flipkart", "cart", "shopping spree", "bargain", "deal", "discount", "sale"
    ]
    if any(keyword in msg for keyword in shopping_keywords):
        return "Shopping"
    
    # Housing: ~20 keywords.
    housing_keywords = [
        "rent", "lease", "mortgage", "house", "apartment", "condo", "home", 
        "dwelling", "residence", "flat", "rental", "housing", "rent payment", 
        "renting", "sublet", "homestay", "lodging", "tenant", "property", "estate"
    ]
    if any(keyword in msg for keyword in housing_keywords):
        return "Housing"
    
    # Insurance: ~20 keywords.
    insurance_keywords = [
        "insurance", "premium", "coverage", "insurer", "policy", "protection", 
        "insured", "claim", "compensation", "risk", "indemnity", "assurance", 
        "liability", "accident", "health insurance", "vehicle insurance", 
        "home insurance", "life insurance", "claim settlement", "insurance premium"
    ]
    if any(keyword in msg for keyword in insurance_keywords):
        return "Insurance"
    
    # Medical: ~20 keywords.
    medical_keywords = [
        "hospital", "clinic", "surgery", "health", "doctor", "emergency", 
        "operation", "medical", "treatment", "diagnosis", "therapy", "care", 
        "ambulance", "inpatient", "outpatient", "consultation", "nurse", "ward", "icu", "medic"
    ]
    if any(keyword in msg for keyword in medical_keywords):
        return "Medical"
    
    return None

def predict_transaction(message):
    # First, try rule-based prediction.
    rule_pred = rule_based_prediction(message)
    if rule_pred is not None:
        return rule_pred

    # Otherwise, use the ML model prediction.
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    mapping = {
        0: "Credited",
        1: "Entertainment",
        2: "Food",
        3: "Other",
        4: "Pharmacy",
        5: "Shopping",
        6: "Transportation",
        7: "Utilities"
    }
    return mapping.get(predicted_index, "Unknown")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/streamlit-page', methods=['GET', 'POST'])
def streamlit_page():
    if request.method == 'POST':
        current_file = request.files['current_file']
        previous_file = request.files['previous_file']
        savings_percentage = float(request.form['savings_percentage'])
        income = float(request.form['income'])
        
        if current_file and previous_file:
            current_month = pd.read_csv(current_file)
            previous_month = pd.read_csv(previous_file)
            
            current_summary = current_month.groupby('Category')['Amount'].sum()
            previous_summary = previous_month.groupby('Category')['Amount'].sum()
            
            combined_summary = pd.DataFrame({
                'Current Month': current_summary,
                'Previous Month': previous_summary
            }).fillna(0)
            
            combined_summary['Change Amount'] = combined_summary['Current Month'] - combined_summary['Previous Month']
            combined_summary['Change Percentage'] = (combined_summary['Change Amount'] / combined_summary['Previous Month']) * 100
            
            total_current = combined_summary['Current Month'].sum()
            total_previous = combined_summary['Previous Month'].sum()
            excess = max(0, total_current - total_previous)
            savings_target = total_current * (savings_percentage / 100)
            
            combined_summary['Excess Reduction'] = (excess * combined_summary['Current Month'] / total_current).round(2)
            combined_summary['Target Reduction'] = (savings_target / len(combined_summary)).round(2)
            combined_summary['Total Reduction'] = combined_summary['Excess Reduction'] + combined_summary['Target Reduction']
            combined_summary['Adjusted Spending'] = combined_summary['Current Month'] - combined_summary['Total Reduction']
            
            pdf_path = os.path.join(STATIC_PATH, "Monthly_Spending_Report.pdf")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Monthly Spending Analysis Report", ln=True, align="C")
            
            for index, row in combined_summary.iterrows():
                pdf.cell(200, 10, txt=f"{index}: Current = {row['Current Month']}, Suggested Reduction = {row['Total Reduction']}, Adjusted = {row['Adjusted Spending']}", ln=True)
            
            pdf.output(pdf_path)
            # Pass HTML version of the summary and PDF path to the template
            return render_template('streamlit.html', data=combined_summary.to_html(classes='table table-striped'), pdf_path=pdf_path)
    
    return render_template('streamlit.html')

@app.route('/download_pdf')
def download_pdf():
    pdf_path = os.path.join(STATIC_PATH, "Monthly_Spending_Report.pdf")
    return send_file(pdf_path, as_attachment=True)

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    prediction = predict_transaction(message)
    return jsonify({"message": message, "prediction": prediction})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
