import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ================= EMAIL CONFIG =================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)

# ================= HEALTH CHECK ROUTE =================
# 👉 ADD THIS HERE (IMPORTANT)
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "Backend is running",
        "service": "TOPSIS API"
    })

# ================= TOPSIS FUNCTION =================
def calculate_topsis(data_matrix, weights, impacts):
    matrix = np.array(data_matrix, dtype=float)
    weights = np.array(weights, dtype=float)

    # Step 1: Normalize
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

    # Step 2: Weighted normalized matrix
    weighted_matrix = norm_matrix * weights

    # Step 3: Ideal best & worst
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])

    for j in range(weighted_matrix.shape[1]):
        if impacts[j] == '+':
            ideal_best[j] = weighted_matrix[:, j].max()
            ideal_worst[j] = weighted_matrix[:, j].min()
        else:
            ideal_best[j] = weighted_matrix[:, j].min()
            ideal_worst[j] = weighted_matrix[:, j].max()

    # Step 4: Distances
    distance_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    # Step 5: TOPSIS Score
    scores = distance_worst / (distance_best + distance_worst)
    return scores.tolist()

# ================= API ROUTE =================
@app.route('/api/topsis/full-process', methods=['POST'])
def full_topsis_process():
    try:
        csv_file = request.files.get('file')
        weights_str = request.form.get('weights')
        impacts_str = request.form.get('impacts')
        recipient_email = request.form.get('email')

        if not all([csv_file, weights_str, impacts_str, recipient_email]):
            return jsonify({'error': 'Missing required fields'}), 400

        # 1. Load CSV
        df = pd.read_csv(csv_file)

        # 2. Extract numeric data (exclude Fund Name)
        data_matrix = df.iloc[:, 1:].values

        # 3. Parse weights & impacts
        weights = [float(w.strip()) for w in weights_str.split(',')]
        impacts = [i.strip() for i in impacts_str.split(',')]

        # 4. TOPSIS calculation
        scores = calculate_topsis(data_matrix, weights, impacts)

        # Attach scores & ranks
        df['Topsis Score'] = scores
        df['Rank'] = df['Topsis Score'].rank(
            ascending=False,
            method='dense'
        ).astype(int)

        result_df = df.copy()

        # ================= EMAIL CSV =================
        csv_buffer = StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # ================= FIND TOP FUND =================
        top_row = result_df.loc[result_df['Rank'] == 1].iloc[0]
        top_fund = top_row['Fund Name']
        top_score = top_row['Topsis Score']

        # ================= SEND EMAIL =================
        msg = Message(
            subject='TOPSIS Analysis: Ranked Results',
            recipients=[recipient_email]
        )

        msg.body = f"""
Hello,

Your TOPSIS analysis is complete.

Top Ranked Alternative: {top_fund}
Score: {top_score:.6f}
Total Alternatives: {len(result_df)}

Attached is the CSV with scores and ranks.

Best regards,
TOPSIS Web Service
"""

        filename = f"topsis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        msg.attach(filename, 'text/csv', csv_content)
        mail.send(msg)

        return jsonify({
            'message': 'Success! Results emailed.',
            'top_result': top_fund
        }), 200

    except Exception as e:
        print("Server Error:", e)
        return jsonify({'error': str(e)}), 500

# ================= RUN =================
if __name__ == '__main__':
    app.run()
