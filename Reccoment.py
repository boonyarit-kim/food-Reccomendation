import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pyodbc
from datetime import datetime

# ==============================
# 1. เชื่อม SQL Server
# ==============================
conn = pyodbc.connect(
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=LAPTOP-HJRRM85F;"
    "Database=FoodRecommendDB;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

# ==============================
# 2. โหลดตารางจาก DB
# ==============================
users = pd.read_sql("SELECT * FROM User_2", conn)
menu_lookup = pd.read_sql("SELECT * FROM menus_2", conn)
df_final = pd.read_csv("df_final2.csv")  
user_ratings = pd.read_sql("SELECT * FROM Ratings", conn)
disease_prep = pd.read_csv(r"preposition\disease_prep.csv")  # โหลด health constraints

# normalize columns
df_final.columns = df_final.columns.str.lower()
menu_lookup.columns = menu_lookup.columns.str.lower()
user_ratings.columns = user_ratings.columns.str.lower()
users.columns = users.columns.str.lower()

# ==============================
# 3. ฟังก์ชันช่วยคำนวณ
# ==============================
def calculate_age(birthdate):
    if pd.isna(birthdate):
        return None
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

def calculate_bmi(weight, height_cm):
    """
    คำนวณ BMI (Body Mass Index)
    weight: น้ำหนัก (kg)
    height_cm: ความสูง (cm)
    return: BMI (float, 2 ตำแหน่ง) หรือ None ถ้าข้อมูลไม่ครบ
    """
    if weight is None or height_cm is None or height_cm == 0:
        return None
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

def calculate_bmr(weight, height_cm, age, gender):
    """
    คำนวณ BMR (Basal Metabolic Rate) โดยใช้สูตร Harris-Benedict
    weight: น้ำหนัก (kg)
    height_cm: ความสูง (cm)
    age: อายุ (ปี)
    gender: 'male' หรือ 'female'
    return: BMR (float, 2 ตำแหน่ง) หรือ None ถ้าข้อมูลไม่ครบ
    """
    if None in [weight, height_cm, age] or height_cm == 0:
        return None
    
    gender = gender.lower() if gender else ''
    
    if gender == 'male':
        bmr = 88.36 + (13.4 * weight) + (4.8 * height_cm) - (5.7 * age)
    else:
        bmr = 447.6 + (9.25 * weight) + (3.1 * height_cm) - (4.33 * age)
    
    return round(bmr, 2)


def get_activity_factor(user_id, conn):
    """
    ดึง activity_factor ของ user จากตาราง activities ตาม activity_id ของ user
    """
    query = f"""
        SELECT a.activity_factor
        FROM user_2 u
        JOIN activities a ON u.activity_id = a.activity_id
        WHERE u.user_id = {user_id}
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return 1.55  # ค่า default ถ้าไม่มีข้อมูล
    return df.iloc[0]['activity_factor']

def calculate_tdee(user_id, bmr, conn):
    """
    คำนวณ TDEE โดยใช้ BMR และ activity_factor ของ user
    """
    if bmr is None:
        return None
    activity_factor = get_activity_factor(user_id, conn)
    return round(bmr * activity_factor, 2)


def get_latest_mood(user_id):
    query = f"""
        SELECT cu.user_id, m.moods
        FROM CurrentUserMood cu
        JOIN Mood m ON cu.mood_id = m.mood_id
        WHERE cu.user_id = {user_id}
        ORDER BY cu.timestamp DESC
    """
    mood_df = pd.read_sql(query, conn)
    if mood_df.empty:
        return None
    return mood_df.iloc[0]['moods'].lower()

# ==============================
# 4. ฟังก์ชัน health constraints
# ==============================
target_diseases = [
    "diabetes", "hypertension", "gout", "obesity", "kidney",
    "dyslipidemia", "heart_disease", "celiac", "lactose_intolerance",
    "anemia", "hyperuricemia"
]

def filter_by_health_constraints(user_id, df_final, df_menu_lookup):
    user_df = users[users['user_id'] == user_id]
    if user_df.empty:
        print(f"⚠️ User ID {user_id} ไม่มีในฐานข้อมูล")
        return df_final
    user = user_df.iloc[0]

    user_diseases = [d for d in target_diseases if d in str(user['disease']).split(',')]

    if not user_diseases:
        return df_final[["food_id"]]

    nutrient_cols = ['food_id','calories',"category",'protein','fat','carbs','sugar','sodium','cholesterol','potassium','saturated_fat']
    df_nutrients = df_menu_lookup[nutrient_cols].copy()
    df_nutrients.columns = [c.lower() for c in df_nutrients.columns]

    df_filtered = df_final[['food_id']].drop_duplicates().merge(df_nutrients, on='food_id', how='left')

    for disease in user_diseases:
        disease_df = disease_prep[disease_prep['disease'] == disease]
        if disease_df.empty:
            print(f"⚠️ ไม่มีข้อมูลโรค {disease} ใน disease_prep")
            continue
        disease_row = disease_df.iloc[0]

        avoid_nutrients = str(disease_row['avoid_nutrients']).split(',')
        for nutrient in avoid_nutrients:
            nutrient = nutrient.strip().lower()
            if nutrient and nutrient != "none" and nutrient in df_filtered.columns:
                if nutrient == "sugar":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 10]
                elif nutrient == "sodium":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 400]
                elif nutrient == "fat":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 20]
                elif nutrient == "calories":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 600]
                elif nutrient == "cholesterol":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 100]
                elif nutrient == "potassium":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 400]
                elif nutrient == "protein":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 20]
                elif nutrient == "saturated_fat":
                    df_filtered = df_filtered[df_filtered[nutrient] <= 10]

        avoid_ingredients = str(disease_row['avoid_ingredients']).split(',')
        for ing in avoid_ingredients:
            ing = ing.strip()
            if ing and ing != "none" and ing in df_final.columns:
                df_filtered = df_filtered[df_final[ing] <= 0.1]

    return df_filtered[["food_id"]]

# ==============================
# 5. ฟังก์ชัน Recommendation
# ==============================
def content_based_recommend(user_id, top_n=5):
    user_df = users[users['user_id'] == user_id]
    if user_df.empty:
        print(f"⚠️ User ID {user_id} ไม่มีในฐานข้อมูล")
        return df_final
    user = user_df.iloc[0]

    # คำนวณ bmi/bmr/tdee
    age = calculate_age(user['birthdate'] if 'birthdate' in user else user['BirthDate'])
    bmi = calculate_bmi(user['weight'], user['height'])
    bmr = calculate_bmr(user['weight'], user['height'], age, user['gender'])
    tdee = calculate_tdee(uid, bmr, conn)
    current_mood = get_latest_mood(user_id)

    # กรอง diet_type
    if user['diet_type'] == 'veg':
        df_filtered = df_final[df_final['veg_non_veg'] == 1].copy()
    else:
        df_filtered = df_final.copy()

    # กรอง dislikes
    dislikes = str(user['dislikes']).split(',')
    for ing in dislikes:
        ing = ing.strip()
        if ing in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[ing] == 0]

    # กรองตาม health constraints
    allowed_foods = set(filter_by_health_constraints(user_id, df_final, menu_lookup)["food_id"].unique())
    df_filtered = df_filtered[df_filtered['food_id'].isin(allowed_foods)]

    # Features
    features = [c for c in df_filtered.columns if c not in ["food_id","food_name"]]
    df_filtered[features] = df_filtered[features].fillna(0)

    # User vector
    user_vector = np.zeros(len(features))
    if "calories" in features:
        tdee_val = tdee if tdee is not None else 2000  # ค่า default 2000 kcal
        user_vector[features.index("calories")] = tdee_val * 0.9
    if "protein" in features:
        user_vector[features.index("protein")] = 20 if bmi >= 25 else 15
    if "fat" in features:
        user_vector[features.index("fat")] = 20 if bmi < 18.5 else 10

    # Mood
    mood_cols = [col for col in features if col in ["happy","neutral","stressed","sad"]]
    for mc in mood_cols:
        user_vector[features.index(mc)] = 0
    if current_mood in mood_cols:
        user_vector[features.index(current_mood)] = 1

    # Likes
    likes = str(user['likes']).split(',')
    for like in likes:
        like = like.strip()
        if like in features:
            user_vector[features.index(like)] = 1

    similarity = cosine_similarity(df_filtered[features].values, [user_vector])
    top_indices = np.argsort(similarity[:,0])[::-1][:top_n]

    food_ids = df_filtered.iloc[top_indices]['food_id'].values
    result = menu_lookup[menu_lookup['food_id'].isin(food_ids)][
        ["food_id","food_name","category","calories","protein","carbs","fat","sugar","sodium"]
    ]
    result = result.set_index('food_id').reindex(food_ids).reset_index()

    return result

def collaborative_recommend(user_id, top_n=5, include_eaten=True):
    if user_ratings.empty:
        return pd.DataFrame()

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(user_ratings[['user_id', 'food_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(random_state=42)
    algo.fit(trainset)

    allowed_foods = set(filter_by_health_constraints(user_id, df_final, menu_lookup)["food_id"].unique())
    all_foods = set(menu_lookup['food_id'].unique())
    eaten_foods = set(user_ratings[user_ratings['user_id'] == user_id]['food_id'].unique())

    if include_eaten:
        candidate_foods = sorted(list(all_foods & allowed_foods))
    else:
        candidate_foods = sorted(list((all_foods - eaten_foods) & allowed_foods))

    if not candidate_foods:
        return pd.DataFrame()

    predictions = [algo.predict(user_id, fid) for fid in candidate_foods]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_pred = predictions[:top_n]
    food_ids = [int(pred.iid) for pred in top_pred]

    result = menu_lookup[menu_lookup['food_id'].isin(food_ids)][
        ["food_id", "food_name", "calories", "protein", "carbs", "fat", "sugar"]
    ]
    result['pred_score'] = [pred.est for pred in top_pred]
    result = result.set_index('food_id').reindex(food_ids).reset_index()

    return result


def hybrid_recommend(user_id, top_n=5, alpha=0.5):
    content_rec = content_based_recommend(user_id, top_n=20)
    collab_rec = collaborative_recommend(user_id, top_n=20)

    if collab_rec.empty:
        return content_rec.head(top_n)[["food_id","food_name","category", "calories", "protein", "carbs", "fat","sugar","sodium"]]

    content_rec = content_rec.copy().sort_values("food_id")
    content_rec["content_score"] = np.arange(len(content_rec), 0, -1)

    collab_rec = collab_rec.copy().sort_values("food_id")
    collab_rec["collab_score"] = np.arange(len(collab_rec), 0, -1)

    merged = pd.concat([content_rec, collab_rec], ignore_index=True)
    merged = merged.groupby("food_name", as_index=False).first()
    merged["content_score"] = merged.get("content_score", 0)
    merged["collab_score"] = merged.get("collab_score", 0)
    merged["hybrid_score"] = alpha * merged["content_score"] + (1 - alpha) * merged["collab_score"]

    return merged.sort_values("hybrid_score", ascending=False).head(top_n)[
        ["food_id","food_name","category", "calories", "protein", "carbs", "fat","sugar",'sodium']
    ]

# ==============================
# 6. ทดสอบระบบ
# ==============================
uid = 52
user_df = users[users["user_id"]==uid].copy()
if not user_df.empty:
    user = user_df.iloc[0]
    age = calculate_age(user['birthdate'] if 'birthdate' in user else user['BirthDate'])
    bmi = calculate_bmi(user['weight'], user['height'])
    bmr = calculate_bmr(user['weight'], user['height'], age, user['gender'])
    tdee = calculate_tdee(uid, bmr, conn)

    current_mood = get_latest_mood(uid)

    user_info = pd.DataFrame([{
        "username": user["username"],
        "gender": user["gender"],
        "age": age,
        "bmi": bmi,
        "bmr": bmr,
        "tdee": tdee,
        "disease": user["disease"],
        "current_mood": current_mood
    }])
    print("👤 User Information:")
    print(user_info.to_string(index=False))

print("\n🎯 Hybrid Recommendation:")
print(hybrid_recommend(uid, top_n=5))
