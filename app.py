# encoding=utf-8
from flask import Flask
from flask import render_template
from flask import request
import xgboost as xgb
import json
import pymysql
import numpy as np
import pandas as pd
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_data()
    data = json.loads(data)
    city = data.get('city')
    brand = data.get('brand')
    output_volume = data.get('output_volume')
    launch_year = data.get('launch_year')
    kilometres = data.get('kilometres')
    is_import = data.get('is_import')
    old_price = data.get('old_price')
    gear_type = data.get('gear_type')
    g1, g2 = 0, 1
    if gear_type == '自动挡':
        g1, g2 = 0, 1
    else:
        g1, g2 = 1, 0
    l = [city, brand, output_volume, launch_year, kilometres, is_import, old_price, g1, g2]
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='root', charset='utf8', database='test')
    cursor = conn.cursor()
    sql = 'select city, brand, output_volume from t_encoder order by id desc limit 1'
    cursor.execute(sql)
    result = cursor.fetchone()
    city_l = np.array(json.loads(result[0]))
    brand_l = np.array(json.loads(result[1]))
    output_volume_l = np.array(json.loads(result[2]))

    re_assign(2, output_volume_l, l)
    re_assign(0, city_l, l)
    re_assign(1, brand_l, l)
    t = []
    for v in l:
        if isinstance(v, str):
            t.append(float(v))
        else:
            t.append(v)

    tu = tuple(t)
    t = list()
    t.append(tu)
    print(pd)
    t = pd.DataFrame(t, columns=['city', 'brand', 'output_volume', 'launch_year', 'kilometres', 'is_import', 'old_price', 'gear_type_手动挡', 'gear_type_自动挡'])

    model = xgb.Booster(model_file='xgboost.model')
    t = xgb.DMatrix(t)
    result = model.predict(t)
    result = float(result[0])
    print(type(result))
    return {'result': result}

@app.route('/add', methods=['GET', 'POST'])
def add():
    return 'success'


# 非数值类型数据转换为数值类型（以原转换为准则）
def re_assign(index, ls, tbd):
    for i in range(len(ls)):
        tbd[index] = i + 1
        if ls[i] == tbd[index]:
            tbd[index] = i
            break