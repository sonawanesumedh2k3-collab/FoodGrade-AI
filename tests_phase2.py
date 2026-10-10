"""Phase 2 quick integration test."""
import sys, os
sys.path.insert(0, r"C:\Users\Sumedh\Desktop\IR Project\FoodGrade_AI_Main")
os.chdir(r"C:\Users\Sumedh\Desktop\IR Project\FoodGrade_AI_Main")

errors = []

# ── 1. Import check ──────────────────────────────────────────
try:
    from services.v6_ml import train_health_model, load_model_metrics, predict_health_grade
    from services.v5_recommender import recommend_similar_products, recommend_healthier_alternatives
    print("[PASS] Imports OK")
except Exception as e:
    errors.append(f"Import failed: {e}")
    print(f"[FAIL] Import: {e}")
    sys.exit(1)

# ── 2. Train model ───────────────────────────────────────────
try:
    result = train_health_model()
    ds = result["dataset_size"]
    acc = result["accuracy"]
    cv = result.get("cross_val_mean", "N/A")
    state = result.get("model_state", "?")
    print(f"[PASS] train_health_model: dataset={ds}, accuracy={acc}%, cv_mean={cv}%, state={state}")
except Exception as e:
    errors.append(f"train_health_model: {e}")
    print(f"[FAIL] train_health_model: {e}")

# ── 3. Load cached metrics ───────────────────────────────────
try:
    cached = load_model_metrics()
    print(f"[PASS] load_model_metrics: state={cached.get('model_state')}, dataset={cached.get('dataset_size')}")
    assert cached.get("model_state") == "cached", f"Expected 'cached', got {cached.get('model_state')}"
except Exception as e:
    errors.append(f"load_model_metrics: {e}")
    print(f"[FAIL] load_model_metrics: {e}")

# ── 4. Prediction ────────────────────────────────────────────
try:
    pred = predict_health_grade({
        "calories": 300, "sugar": 3, "total_fat": 5, "saturated_fat": 1,
        "protein": 14, "fibre": 10, "salt": 0.2
    })
    grade = pred.get("grade")
    conf = pred.get("confidence", {})
    print(f"[PASS] predict_health_grade: grade={grade}, confidence_keys={list(conf.keys())}")
    assert grade in ("A","B","C","D","E"), f"Bad grade: {grade}"
except Exception as e:
    errors.append(f"predict_health_grade: {e}")
    print(f"[FAIL] predict_health_grade: {e}")

# ── 5. Recommend similar ─────────────────────────────────────
p1 = {"id":1,"product_name":"Oats Cereal","brand":"A","category":"Cereal",
      "ingredients":"oats","claims":"high fibre","health_grade":"A","health_score":85}
p2 = {"id":2,"product_name":"Chocolate Biscuit","brand":"B","category":"Biscuit",
      "ingredients":"flour sugar","claims":"","health_grade":"D","health_score":40}
p3 = {"id":3,"product_name":"Quinoa Cereal","brand":"C","category":"Cereal",
      "ingredients":"quinoa oats","claims":"protein rich","health_grade":"A","health_score":90}

try:
    similar = recommend_similar_products(p1, [p1,p2,p3], top_n=5)
    print(f"[PASS] recommend_similar_products: {len(similar)} result(s)")
    # p1 itself must be excluded
    names = [r["product"]["product_name"] for r in similar]
    assert "Oats Cereal" not in names or len(names) == 0, "Target product leaked into results"
except Exception as e:
    errors.append(f"recommend_similar_products: {e}")
    print(f"[FAIL] recommend_similar_products: {e}")

# ── 6. Healthier alternatives ────────────────────────────────
try:
    healthier = recommend_healthier_alternatives(p1, [p1,p2,p3], top_n=3)
    print(f"[PASS] recommend_healthier_alternatives: {len(healthier)} result(s)")
    for h in healthier:
        hg = h.get("health_grade")
        hs = h.get("health_score", 0)
        # Must be better than p1 (A, 85) — same grade A but higher score, or nothing
        print(f"  -> {h['product']['product_name']}: grade={hg}, score={hs}")
except Exception as e:
    errors.append(f"recommend_healthier_alternatives: {e}")
    print(f"[FAIL] recommend_healthier_alternatives: {e}")

# ── 7. Empty vocab guard ─────────────────────────────────────
try:
    empty_p = {"id":99,"product_name":"","brand":"","category":"","ingredients":"","claims":"","health_grade":"C","health_score":55}
    result2 = recommend_similar_products(empty_p, [empty_p, p1], top_n=3)
    print(f"[PASS] Empty-vocab guard: returned {len(result2)} result(s) without crash")
except Exception as e:
    errors.append(f"empty_vocab_guard: {e}")
    print(f"[FAIL] Empty-vocab guard: {e}")

# ── Summary ──────────────────────────────────────────────────
print()
if errors:
    print(f"PHASE 2 TEST FAILED — {len(errors)} error(s):")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("PHASE 2 ALL TESTS PASSED")
