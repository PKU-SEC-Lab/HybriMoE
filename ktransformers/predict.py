import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def parse_log(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    X = []
    y = []

    for key, value in data.items():
        if key == "fin":
            continue
        expert_num = int(key[0])
        token_num = int(key[1:])
        avg_time = value["avg"]

        X.append([expert_num, token_num])
        y.append(avg_time)

    return np.array(X), np.array(y)


# 拟合多项式表达式
def fit_polynomial(X, y, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly


# 生成多项式表达式
def generate_expression(model, poly):
    terms = []
    intercept = model.intercept_
    terms.append(f"{intercept:.12f}")
    for i, coef in enumerate(model.coef_[1:], 1):
        term_parts = []
        for j, exp in enumerate(poly.powers_[i]):
            if exp != 0:
                if j == 0:
                    if exp == 1:
                        term_parts.append("{expert_num}")
                    else:
                        term_parts.append("({expert_num} ** " + str(exp) + ")")
                elif j == 1:
                    if exp == 1:
                        term_parts.append("{token_num}")
                    else:
                        term_parts.append("({token_num} ** " + str(exp) + ")")
        term = " * ".join(term_parts)
        if term:
            terms.append(f"{coef:.12f} * {term}")
    expression = " + ".join(terms)
    return expression


# 主函数
def main():
    log_file_path = "deepseek_prefill_time.log"

    X, y = parse_log(log_file_path)

    model, poly = fit_polynomial(X, y, degree=3)

    expression = generate_expression(model, poly)
    print(f"exp: {expression}")

    expert_nums = [1, 2, 3, 4]
    token_nums = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for expert_num in expert_nums:
        for token_num in token_nums:
            exp = expression.format(expert_num=expert_num, token_num=token_num)
            time = eval(exp)
            t2 = model.predict(poly.fit_transform([[expert_num, token_num]]))[0]
            print(abs(time - t2)) # 误差
            print(f"expert_num: {expert_num}, token_num: {token_num}, time: {time}")
            real_time = y[np.where((X == [expert_num, token_num]).all(axis=1))][0]
            print(f"Real time: {real_time}, Diff: {time - real_time}")


if __name__ == "__main__":
    main()
