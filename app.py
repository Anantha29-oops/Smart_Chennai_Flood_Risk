from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
from pathlib import Path
import os


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# Smart_Chennai_Flood_Risk/
#
# app/
#     app.py
#
# models/
# data/
#
# This works on both Windows and Render/Linux.

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "random_forest":
        MODEL_DIR / "random_forest_flood_risk.pkl",

    "gradient_boosting":
        MODEL_DIR / "gradient_boosting_flood_risk.pkl",

    "xgboost":
        MODEL_DIR / "xgboost_flood_risk.pkl"
}


FEATURE_FILE = MODEL_DIR / "model_features.pkl"


# ============================================================
# WARD DATA
# ============================================================

WARD_FILE = (
    DATA_DIR /
    "chennai_ward_flood_features.csv"
)


# ============================================================
# RISK CLASS NAMES
# ============================================================

RISK_NAMES = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Very High"
}


# ============================================================
# APPLICATION STARTUP
# ============================================================

print("=" * 70)
print("SMART CHENNAI FLOOD RISK ANALYTICS")
print("STARTING FLASK BACKEND")
print("=" * 70)

print()
print("BASE DIRECTORY:")
print(BASE_DIR)

print()
print("MODEL DIRECTORY:")
print(MODEL_DIR)

print()
print("WARD FILE:")
print(WARD_FILE)


# ============================================================
# LOAD MODELS
# ============================================================

models = {}


for model_name, model_path in MODEL_FILES.items():

    try:

        if not model_path.exists():

            print(
                f"WARNING - {model_name}: "
                f"model file not found"
            )

            continue

        models[model_name] = joblib.load(
            model_path
        )

        print(
            f"OK - {model_name}: "
            f"{type(models[model_name]).__name__}"
        )

    except Exception as e:

        print(
            f"ERROR - {model_name}: {e}"
        )


# ============================================================
# LOAD FEATURE LIST
# ============================================================

FEATURES = []


try:

    if not FEATURE_FILE.exists():

        print(
            "WARNING - model_features.pkl "
            "not found"
        )

    else:

        FEATURES = joblib.load(
            FEATURE_FILE
        )

        print(
            f"OK - Model features loaded: "
            f"{len(FEATURES)}"
        )

except Exception as e:

    print(
        "ERROR loading model features:",
        e
    )


# ============================================================
# LOAD WARD DATA
# ============================================================

wards = pd.DataFrame()


try:

    if not WARD_FILE.exists():

        print(
            "ERROR - Ward dataset not found:"
        )

        print(WARD_FILE)

    else:

        wards = pd.read_csv(
            WARD_FILE
        )

        print()
        print("OK - Ward dataset loaded")
        print(
            f"Shape: {wards.shape}"
        )
        print(
            f"Rows: {len(wards)}"
        )
        print(
            f"Columns: {list(wards.columns)}"
        )

except Exception as e:

    print(
        "ERROR loading ward dataset:",
        e
    )


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

if not wards.empty:

    wards.columns = (
        wards.columns
        .astype(str)
        .str.strip()
    )


# ============================================================
# WARD DATA CHECK
# ============================================================

print()
print("=" * 70)
print("WARD DATA CHECK")
print("=" * 70)

required_columns = [
    "ward_id",
    "ward",
    "zone",
    "region",
    "flood_probability",
    "risk_class"
]

for column in required_columns:

    if column in wards.columns:

        print(f"OK   : {column}")

    else:

        print(f"MISS : {column}")

print("=" * 70)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):

    """
    Convert pandas/numpy values into
    JSON-friendly values.
    """

    if pd.isna(value):

        return None

    try:

        return float(value)

    except (TypeError, ValueError):

        return str(value)


def get_risk_class_from_probability(probability):

    """
    Used only when risk_class is unavailable.
    """

    try:

        probability = float(probability)

    except:

        return "Unknown"

    if probability < 25:

        return "Low"

    elif probability < 50:

        return "Moderate"

    elif probability < 75:

        return "High"

    else:

        return "Very High"


def get_ward_by_id_or_name(identifier):

    """
    Find a ward using ward_id or ward name.
    """

    if wards.empty:

        return None

    identifier = str(identifier).strip()

    # First try ward_id
    if "ward_id" in wards.columns:

        matches = wards[
            wards["ward_id"]
            .astype(str)
            .str.strip()
            == identifier
        ]

        if not matches.empty:

            return matches.iloc[0]

    # Then try ward name
    if "ward" in wards.columns:

        matches = wards[
            wards["ward"]
            .astype(str)
            .str.strip()
            .str.lower()
            == identifier.lower()
        ]

        if not matches.empty:

            return matches.iloc[0]

    return None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# API: HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "running",

        "models_loaded":
            list(models.keys()),

        "feature_count":
            len(FEATURES),

        "ward_count":
            len(wards)
            if not wards.empty
            else 0

    })


# ============================================================
# API: SUMMARY
# ============================================================

@app.route("/api/summary")
def get_summary():

    try:

        if wards.empty:

            return jsonify({
                "error":
                    "Ward dataset is not loaded."
            }), 500

        total_wards = len(wards)

        # ----------------------------------------------------
        # FLOOD PROBABILITY
        # ----------------------------------------------------

        if "flood_probability" in wards.columns:

            probabilities = pd.to_numeric(
                wards["flood_probability"],
                errors="coerce"
            ).dropna()

        else:

            probabilities = pd.Series(
                dtype=float
            )


        if not probabilities.empty:

            average_probability = float(
                probabilities.mean()
            )

            max_probability = float(
                probabilities.max()
            )

        else:

            average_probability = 0.0

            max_probability = 0.0


        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        distribution = {
            "Low": 0,
            "Moderate": 0,
            "High": 0,
            "Very High": 0
        }


        if "risk_class" in wards.columns:

            for value in wards[
                "risk_class"
            ].dropna():

                risk = str(value).strip()

                if risk in distribution:

                    distribution[risk] += 1


        return jsonify({

            "total_wards":
                total_wards,

            "average_probability":
                average_probability,

            "max_probability":
                max_probability,

            "model_features":
                len(FEATURES),

            "risk_distribution":
                distribution

        })

    except Exception as e:

        print(
            "Summary error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# API: AVAILABLE MODELS
# ============================================================

@app.route("/api/models")
def get_models():

    available_models = []


    if "random_forest" in models:

        available_models.append({

            "id":
                "random_forest",

            "name":
                "Random Forest"

        })


    if "gradient_boosting" in models:

        available_models.append({

            "id":
                "gradient_boosting",

            "name":
                "Gradient Boosting"

        })


    if "xgboost" in models:

        available_models.append({

            "id":
                "xgboost",

            "name":
                "XGBoost"

        })


    return jsonify({

        "models":
            available_models

    })


# ============================================================
# API: AVAILABLE WARDS
# ============================================================

@app.route("/api/wards")
def get_wards():

    try:

        if wards.empty:

            return jsonify({
                "error":
                    "Ward dataset is not loaded."
            }), 500


        ward_list = []


        for _, row in wards.iterrows():

            ward_id = row.get(
                "ward_id",
                ""
            )

            ward_name = row.get(
                "ward",
                ""
            )

            zone = row.get(
                "zone",
                ""
            )

            region = row.get(
                "region",
                ""
            )

            probability = row.get(
                "flood_probability",
                0
            )

            risk_class = row.get(
                "risk_class",
                None
            )


            probability = clean_value(
                probability
            )


            if risk_class is None:

                risk_class = (
                    get_risk_class_from_probability(
                        probability
                    )
                )

            else:

                risk_class = str(
                    risk_class
                )


            ward_list.append({

                "ward_id":
                    str(ward_id),

                "ward":
                    str(ward_name),

                "zone":
                    str(zone),

                "region":
                    str(region),

                "flood_probability":
                    probability,

                "risk_class":
                    risk_class

            })


        return jsonify(
            ward_list
        )

    except Exception as e:

        print(
            "Ward API error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# API: WARD INFORMATION
# ============================================================

@app.route("/api/ward/<path:ward_id>")
def get_ward(ward_id):

    try:

        if wards.empty:

            return jsonify({
                "error":
                    "Ward dataset is not loaded."
            }), 500


        row = get_ward_by_id_or_name(
            ward_id
        )


        if row is None:

            return jsonify({

                "error":
                    f"Ward {ward_id} not found."

            }), 404


        data = {}


        for column in wards.columns:

            if column == "geometry":

                continue


            data[column] = clean_value(
                row[column]
            )


        # ----------------------------------------------------
        # RISK CLASS
        # ----------------------------------------------------

        probability = data.get(
            "flood_probability"
        )


        if not data.get(
            "risk_class"
        ):

            data["risk_class"] = (
                get_risk_class_from_probability(
                    probability
                )
            )


        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        risk_class = str(
            data.get(
                "risk_class",
                "Unknown"
            )
        )


        messages = {

            "Low":
                "This ward currently shows a relatively low flood-risk level based on the available flood and spatial indicators.",

            "Moderate":
                "This ward shows a moderate level of flood risk. Drainage and flood-prone areas should be monitored.",

            "High":
                "This ward shows a high level of flood risk. Drainage management and flood preparedness should receive increased attention.",

            "Very High":
                "This ward shows a very high level of flood risk. Strong flood mitigation and preparedness measures are recommended."

        }


        data["risk_message"] = messages.get(
            risk_class,
            "Flood-risk information is available for this ward."
        )


        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        recommendations = []


        if "recommendations" in wards.columns:

            recommendation_text = row[
                "recommendations"
            ]


            if not pd.isna(
                recommendation_text
            ):

                recommendations = [

                    item.strip()

                    for item in str(
                        recommendation_text
                    ).split(";")

                    if item.strip()

                ]


        data["recommendations"] = (
            recommendations
        )


        return jsonify(data)


    except Exception as e:

        print(
            "Ward information error:",
            e
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# API: PREDICT WARD
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        request_data = (
            request.get_json()
        )


        if not request_data:

            return jsonify({

                "error":
                    "No JSON data received."

            }), 400


        ward_id = request_data.get(
            "ward_id"
        )


        model_name = request_data.get(
            "model",
            "random_forest"
        )


        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if model_name not in models:

            return jsonify({

                "error":
                    f"Model '{model_name}' is not available."

            }), 400


        # ----------------------------------------------------
        # CHECK WARD
        # ----------------------------------------------------

        ward = get_ward_by_id_or_name(
            ward_id
        )


        if ward is None:

            return jsonify({

                "error":
                    f"Ward {ward_id} not found."

            }), 404


        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in FEATURES

            if feature not in wards.columns

        ]


        if missing_features:

            return jsonify({

                "error":
                    "Missing model features.",

                "missing_features":
                    missing_features

            }), 500


        # ----------------------------------------------------
        # CREATE FEATURE VECTOR
        # ----------------------------------------------------

        feature_values = []


        for feature in FEATURES:

            value = ward[feature]


            if pd.isna(value):

                value = 0


            feature_values.append(
                float(value)
            )


        X = pd.DataFrame(

            [feature_values],

            columns=FEATURES

        )


        # ----------------------------------------------------
        # SELECT MODEL
        # ----------------------------------------------------

        model = models[
            model_name
        ]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )


        risk_class = RISK_NAMES.get(
            prediction,
            "Unknown"
        )


        # ----------------------------------------------------
        # MODEL PROBABILITIES
        # ----------------------------------------------------

        probabilities = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(X)[0]
            )


        # ----------------------------------------------------
        # EXISTING WARD PROBABILITY
        # ----------------------------------------------------

        existing_probability = None


        if (
            "flood_probability"
            in wards.columns
        ):

            value = ward[
                "flood_probability"
            ]


            if not pd.isna(value):

                existing_probability = (
                    float(value)
                )


        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        recommendations = []


        if (
            "recommendations"
            in wards.columns
        ):

            recommendation_text = ward[
                "recommendations"
            ]


            if not pd.isna(
                recommendation_text
            ):

                recommendations = [

                    item.strip()

                    for item in str(
                        recommendation_text
                    ).split(";")

                    if item.strip()

                ]


        # ----------------------------------------------------
        # MODEL NAME
        # ----------------------------------------------------

        model_names = {

            "random_forest":
                "Random Forest",

            "gradient_boosting":
                "Gradient Boosting",

            "xgboost":
                "XGBoost"

        }


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        result = {

            "success":
                True,

            "ward": {

                "ward_id":
                    str(
                        ward.get(
                            "ward_id",
                            ""
                        )
                    ),

                "ward":
                    str(
                        ward.get(
                            "ward",
                            ""
                        )
                    ),

                "zone":
                    str(
                        ward.get(
                            "zone",
                            ""
                        )
                    ),

                "region":
                    str(
                        ward.get(
                            "region",
                            ""
                        )
                    )

            },

            "model": {

                "id":
                    model_name,

                "name":
                    model_names.get(
                        model_name,
                        model_name
                    )

            },

            "prediction": {

                "class_id":
                    prediction,

                "risk_class":
                    risk_class

            },

            "flood_probability":
                existing_probability,

            "model_probabilities": (

                probabilities.tolist()

                if probabilities
                is not None

                else None

            ),

            "recommendations":
                recommendations

        }


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "Prediction error:",
            e
        )


        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "error":
            "Requested resource was not found."

    }), 404


@app.errorhandler(500)
def internal_error(error):

    return jsonify({

        "error":
            "Internal server error."

    }), 500


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SERVER READY")
    print("=" * 70)

    print(
        "Models:",
        list(models.keys())
    )

    print(
        "Features:",
        len(FEATURES)
    )

    print(
        "Wards:",
        len(wards)
        if not wards.empty
        else 0
    )

    print()
    print(
        "Local URL:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()
    print(
        "Health check:"
    )

    print(
        "http://127.0.0.1:5000/api/health"
    )

    print()
    print(
        "Ward API:"
    )

    print(
        "http://127.0.0.1:5000/api/wards"
    )

    print("=" * 70)


    # Render provides PORT through
    # an environment variable.
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )