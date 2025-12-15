from prediction import predict

def main():
    input_data = {
        "Age": 35,
        "MonthlyCharges": 70.5,
        "Contract": 1
    }

    prediction, confidence = predict(input_data)

    print("Hasil Prediksi")
    print("--------------")
    print("Prediction:", prediction)
    print("Confidence:", round(confidence, 2))


if __name__ == "__main__":
    main()
