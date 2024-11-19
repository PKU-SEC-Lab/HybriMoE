import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# 解析日志数据
def parse_log(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    token_expert_time = {}

    for key, value in data.items():
        expert_num = int(key[0])
        token_num = int(key[1:])
        avg_time = value["avg"]

        if token_num not in token_expert_time:
            token_expert_time[token_num] = {"expert_nums": [], "times": []}

        token_expert_time[token_num]["expert_nums"].append(expert_num)
        token_expert_time[token_num]["times"].append(avg_time)

    return token_expert_time


# 拟合每种 token 数的专家数和时间的一次函数
def fit_linear_functions(token_expert_time):
    token_functions = {}

    for token_num, data in token_expert_time.items():
        X = np.array(data["expert_nums"]).reshape(-1, 1)
        y = np.array(data["times"])

        model = LinearRegression()
        model.fit(X, y)

        coef = model.coef_[0]
        intercept = model.intercept_

        token_functions[token_num] = (coef, intercept)

    return token_functions


# 拟合 coef 和 intercept 关于 token_num 的函数
def fit_coef_intercept_functions(token_functions, degree=2):
    token_nums = np.array(sorted(token_functions.keys()))
    coefs = np.array([token_functions[token_num][0] for token_num in token_nums])
    intercepts = np.array([token_functions[token_num][1] for token_num in token_nums])

    poly = PolynomialFeatures(degree)
    token_nums_poly = poly.fit_transform(token_nums.reshape(-1, 1))

    coef_model = LinearRegression()
    coef_model.fit(token_nums_poly, coefs)

    intercept_model = LinearRegression()
    intercept_model.fit(token_nums_poly, intercepts)

    return coef_model, intercept_model, poly


# 生成函数表达式
def generate_expression(model, poly):
    terms = []
    intercept = model.intercept_
    terms.append(f"{intercept:.12f}")
    for i, coef in enumerate(model.coef_[1:], 1):
        term_parts = []
        for j, exp in enumerate(poly.powers_[i]):
            if exp != 0:
                term_parts.append("({token_num} ** " + str(exp) + ")")
        term = " * ".join(term_parts)
        if term:
            terms.append(f"{coef:.12f} * {term}")
    expression = " + ".join(terms)
    return expression


# 生成函数，根据 token 数给出对应的专家数和时间的一次函数
def generate_function(coef_model, intercept_model, poly):
    def predict(token_num, expert_num):
        token_num_poly = poly.transform(np.array([[token_num]]))
        coef = coef_model.predict(token_num_poly)[0]
        intercept = intercept_model.predict(token_num_poly)[0]
        return coef * expert_num + intercept

    return predict


# 主函数
def main():
    log_file_path = "deepseek_prefill_time.log"

    # 解析日志数据
    token_expert_time = parse_log(log_file_path)

    # 拟合每种 token 数的专家数和时间的一次函数
    token_functions = fit_linear_functions(token_expert_time)

    # 拟合 coef 和 intercept 关于 token_num 的函数
    coef_model, intercept_model, poly = fit_coef_intercept_functions(token_functions, degree=2)

    # 打印 coef 和 intercept 关于 token_num 的函数表达式
    coef_expression = generate_expression(coef_model, poly)
    intercept_expression = generate_expression(intercept_model, poly)
    print(f"Coef function: {coef_expression}")
    print(f"Intercept function: {intercept_expression}")
    print(eval(intercept_expression.format(token_num=64)))
    # 生成函数，根据 token 数给出对应的专家数和时间的一次函数
    predict = generate_function(coef_model, intercept_model, poly)

    # 示例使用
    # token_nums = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # expert_nums = [1, 2, 3, 4, 5, 6]
    # for token_num in token_nums:
    #     for expert_num in expert_nums:
    #         predicted_time = predict(token_num, expert_num)
    #         print(f"Predicted time for {expert_num} experts and {token_num} tokens: {predicted_time:.2f} ms")
    #         diff = (
    #             token_expert_time[token_num]["times"][token_expert_time[token_num]["expert_nums"].index(expert_num)]
    #             - predicted_time
    #         )
    #         print(f"Diff: {diff:.2f} ms")

    # 测试任意整数的 token 数
    # token_num = 1500
    # predicted_time = predict(token_num, expert_num)
    # print(f"Predicted time for {expert_num} experts and {token_num} tokens: {predicted_time:.2f} ms")


if __name__ == "__main__":
    main()
