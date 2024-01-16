import os
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import timedelta


nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'diagnosis.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

suggestions_dict = {
    "Influenza": {
        "advice": "Should not attend work, school, or daycare. Wash hands frequently by using soap and warm water for 15-20 seconds. Drink plenty of fluids, rest and take over-the-counter pain relievers.",
        "doctors": [
            {"name": "Dr. Sumon", "hospital": "City Hospital, Street ABC", "contact": "+1234567890","location":"https://www.google.com/maps?s=web&lqi=ChdtYXBzIEluZmx1ZW56YSBob3NwaXRhbEiC2pKB5oCAgAhaIRAAEAEQAhgCIhdtYXBzIGluZmx1ZW56YSBob3NwaXRhbJIBEHByaXZhdGVfaG9zcGl0YWyqAXEKCC9tLzA0X3RiCggvbS8waHBuchABKhsiF21hcHMgaW5mbHVlbnphIGhvc3BpdGFsKAAyHxABIhtCmNhmd5zPwPPr1DDW9aIwzyOJsqT8QvLybwcyGxACIhdtYXBzIGluZmx1ZW56YSBob3NwaXRhbA&vet=12ahUKEwjyncf2ls6BAxXFTWwGHYJNBicQ1YkKegQIFxAB..i&cs=0&um=1&ie=UTF-8&fb=1&gl=bd&sa=X&geocode=Kb-LkhxMv1U3MR0WTBs6PMiB&daddr=P9WC%2B84Q,+Rd+No+9A,+Dhaka+1209"},
            {"name": "Dr. Sarmin", "hospital": "Health Clinic, Street XYZ", "contact": "+0987654321","location":"https://www.google.com/maps/dir/23.7549689,90.3590926/Emergency+Department,+Bangladesh+Medical+Hospital,+14%2FA,+Dhaka+1209/@23.7535651,90.3630032,15z/data=!3m1!4b1!4m19!1m8!3m7!1s0x3755bf85feb609ab:0x7fb5ec036dc9576d!2sEmergency+Department,+Bangladesh+Medical+Hospital!8m2!3d23.7500915!4d90.3693351!15sChdpbmZsdWVuemEgaG9zcGl0YWwgbWFwc5IBDm1lZGljYWxfY2xpbmlj4AEA!16s%2Fg%2F11pd0j0xj5!4m9!1m1!4e1!1m5!1m1!1s0x3755bf85feb609ab:0x7fb5ec036dc9576d!2m2!1d90.3693363!2d23.7500897!3e0?entry=ttu"}
        ]
    },
    "Migraine": {
        "advice": "Rest in a quiet, dark room and apply a cold compress to your forehead.",
        "doctors": [
            {"name": "Dr. Ria", "hospital": "Neurology Center, Street DEF", "contact": "+8802334455","location":"6,90.359063/Neurology+specialist:+Dr.+Aminur+Rahman,+MD+FACP(USA)+MBBS,+122+Kazi+Nazrul+Islam+Ave,+Dhaka+1000/@23.7492298,90.3353769,13z/data=!3m1!4b1!4m13!1m2!2m1!1smigraine+hospital+bd!4m9!1m1!4e1!1m5!1m1!1s0x3755b894bf6661c1:0xfd7dc1af9f17413b!2m2!1d90.3964821!2d23.7388438!3e0?entry=ttu"},
            {"name": "Dr. Sumona", "hospital": "General Hospital, Street LMN", "contact": "+5566778899","location":"https://www.google.com/maps/dir/23.754956,90.359063/National+Institute+of+Neuro+Sciences+%26+Hospital,+Dhaka+1207/@23.7651897,90.3647682,15z/data=!3m1!4b1!4m19!1m8!3m7!1s0x3755c0b217c31f9b:0xbd96ef36d9e85ce7!2sNational+Institute+of+Neuro+Sciences+%26+Hospital!8m2!3d23.7761345!4d90.3707896!15sChRtaWdyYWluZSBob3NwaXRhbCBiZJIBE2dvdmVybm1lbnRfaG9zcGl0YWzgAQA!16s%2Fg%2F11c7p90__6!4m9!1m1!4e1!1m5!1m1!1s0x3755c0b217c31f9b:0xbd96ef36d9e85ce7!2m2!1d90.3707896!2d23.7761345!3e0?entry=ttu"}
        ]
    },
    "Dengue": {
    "advice": "Reduce Mosquito Habitat. Drink lots of fluids to stay hydrated and take pain relievers with acetaminophen",
    "doctors": [
        {"name": "Dr. Habuilla", "hospital": "Neurology Center, Street DEF", "contact": "+8802334455","location":"https://www.google.com/maps/dir/23.7509415,90.3654296/Bangladesh+Eye+Hospital+%26+Institute,+78+Satmasjid+Road,+Dhaka+1209/@23.7507864,90.3654636,18z/data=!3m1!4b1!4m10!4m9!1m1!4e1!1m5!1m1!1s0x3755bf503a4f31e3:0xa5dcb5caddba4e33!2m2!1d90.3672494!2d23.7516763!3e0?entry=ttu"},
        {"name": "Dr. Begum Banu", "hospital": "General Hospital, Street LMN", "contact": "+5566778899","location":"https://www.google.com/maps"}
    ]
    },
    "COVID-19": {
    "advice": "Stay at home and self-isolate from others. Drink lots of water, rest, and monitor your symptoms. Seek medical attention if you have trouble breathing or persistent pain in the chest",
    "doctors": [
        {"name": "Dr. Cov", "hospital": "Covid Center, Street DEF", "contact": "+8802334455","location":"https://www.google.com/maps/place/DNCC+Dedicated+Covid-19+Hospital,+Mohakhali,+Dhaka-1212/@23.7743561,90.399743,18z/data=!4m5!3m4!1s0x3755c70167b1aff5:0xdc1b3c1531fea45f!8m2!3d23.7742569!4d90.4008692?shorturl=1"},
        {"name": "Dr. Uodian", "hospital": "Vidon Hospital, Street LMN", "contact": "+5566778899","location":"https://www.google.com.bd/maps/@23.8043848,90.4154625,17.5z"}
    ]
    },
 }

class User(UserMixin, db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.set_password(password)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class DiagnosticHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    diagnosis = db.Column(db.String(150), nullable=False)
    text_input = db.Column(db.String(500), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    prompt_message = "Enter your symptoms to get a diagnosis."
    if request.method == 'POST':
        text = request.form['text']
        detected_keywords = detect_keywords(text)
        result = diagnose(detected_keywords)

        if result == "Unknown":
            prompt_message = "We couldn't determine a possible illness based on the provided symptoms. Please provide more detailed information."
        else:
            record = DiagnosticHistory(user_id=current_user.id, diagnosis=result, text_input=text)
            db.session.add(record)
            db.session.commit()
            return redirect(url_for('result'))

    search_query = request.args.get('query', None)
    if search_query:
        user_history = DiagnosticHistory.query.filter(DiagnosticHistory.text_input.ilike(f"%{search_query}%"), 
                                                     DiagnosticHistory.user_id == current_user.id).all()
    else:
        user_history = DiagnosticHistory.query.filter_by(user_id=current_user.id).all()
    return render_template('index.html', doctor_diagnosis=prompt_message, histories=user_history)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        new_user = User(username=username, password=password)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

        if user and user.check_password(password):
            login_user(user, remember=True) 
            return redirect(url_for('index'))
        else:
            error = "Your username or password is incorrect!"
    return render_template('login.html', error=error)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def detect_keywords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    keywords = [w for w in word_tokens if w.isalnum() and w.lower() not in stop_words]
    return keywords

def diagnose(keywords):
    influenza_symptoms = {"fever", "chills", "cough", "throat", "nose", "muscle", "headaches", "fatigue"}
    dengue_symptoms = {"fever", "headache", "pain", "eyes", "joint", "muscle", "rash", "bleeding"}
    migraine_symptoms = {"headache", "nausea", "vomiting", "sensitivity", "light", "sound", "aura"}
    covid19_symptoms = {"fever", "cough", "tiredness", "difficulty", "breathing", "chest", "loss", "taste", "throat"}

    detected_influenza_symptoms = len(set(keywords) & influenza_symptoms)
    detected_migraine_symptoms = len(set(keywords) & migraine_symptoms)
    detected_dengue_symptoms = len(set(keywords) & dengue_symptoms)
    detected_covid19_symptoms = len(set(keywords) & covid19_symptoms)

    possible_diseases = { "Influenza": detected_influenza_symptoms, "Migraine": detected_migraine_symptoms, "Dengue": detected_dengue_symptoms, "COVID-19": detected_covid19_symptoms }

    if max(possible_diseases.values()) == 0:
        return "Unknown"

    most_likely_disease = max(possible_diseases, key=possible_diseases.get)
    return most_likely_disease


@app.route('/result')
@login_required
def result():
    last_record = DiagnosticHistory.query.filter_by(user_id=current_user.id).order_by(DiagnosticHistory.id.desc()).first()
    
    if last_record:
        if last_record.diagnosis == "Unknown":
            result_text = "We couldn't determine a possible illness based on the provided symptoms. It might be helpful to provide more detailed information."
            suggestion = "If you're feeling unwell, always consult with a healthcare professional."
            doctors = []
        else:
            result_text = f"Based on the provided text, there's a higher possibility you might have {last_record.diagnosis}."
            suggestion = suggestions_dict.get(last_record.diagnosis, {}).get("advice", "However, always consult with a healthcare professional regarding health concerns.")
            doctors = suggestions_dict.get(last_record.diagnosis, {}).get("doctors", [])
    else:
        result_text = "No diagnosis available."
        suggestion = ""
        doctors = []


    search_query = request.args.get('query', None)
    if search_query:
        histories = DiagnosticHistory.query.filter(DiagnosticHistory.text_input.ilike(f"%{search_query}%"), 
                                                  DiagnosticHistory.user_id == current_user.id).all()
    else:
        histories = DiagnosticHistory.query.filter_by(user_id=current_user.id).all()

    return render_template('result.html', result=result_text, suggestion=suggestion, histories=histories, doctors=doctors)

@app.route('/history-detail/<int:history_id>', methods=['GET'])
@login_required
def history_detail(history_id):
    record = DiagnosticHistory.query.get(history_id)
    if record and record.user_id == current_user.id:
        return render_template('history_detail.html', record=record)
    return redirect(url_for('index'))

@app.route('/initdb')
def initdb():
    db.create_all()
    return "Database initialized!"


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

