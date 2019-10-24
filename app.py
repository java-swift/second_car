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
import torch
import db
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict_xgb', methods=['GET', 'POST'])
def predict_xg():
    city, brand, output_volume, launch_year, kilometres, is_import, old_price, gear_type = parse_data(request)
    g1, g2 = 0, 1
    if gear_type == '自动挡':
        g1, g2 = 0, 1
    else:
        g1, g2 = 1, 0
    l = [city, brand, output_volume, launch_year, kilometres, is_import, old_price, g1, g2]
    result = get_encoder()
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


@app.route('/predict_torch', methods=['GET', 'POST'])
def predict_torch():
    city, brand, output_volume, launch_year, kilometres, is_import, old_price, gear_type = parse_data(request)
    g1 = 1
    if gear_type == '自动挡': # 自动挡0，手动挡1
        g1 = 0
    else:
        g1 = 1
    result = get_encoder()
    a = [city, brand, output_volume, launch_year, kilometres, is_import, old_price, g1]
    city_l = np.array(json.loads(result[0]))
    brand_l = np.array(json.loads(result[1]))
    re_assign(0, city_l, a)
    re_assign(1, brand_l, a)
    t = []
    for v in a:
        if isinstance(v, str):
            t.append(float(v))
        else:
            t.append(v)
    a = np.array(t)
    result = get_statistics()
    mean_ = np.array(json.loads(result[0]))
    var_ = np.array(json.loads(result[1]))
    b = (a - mean_)/np.sqrt(var_)
    b = torch.from_numpy(b).float()
    model = torch.load('torch.model')
    y_b = model(b).detach().numpy()
    print(y_b)
    return {'result': float(y_b[0])}

# 非数值类型数据转换为数值类型（以原转换为准则）
def re_assign(index, ls, tbd):
    for i in range(len(ls)):
        tbd[index] = i + 1
        if ls[i] == tbd[index]:
            tbd[index] = i
            break


def get_cursor():
    c = db.config()
    conn = pymysql.connect(host=c['host'], port=c['port'], user=c['user'], password=c['password'], charset=c['charset'], database=c['database'])
    cursor = conn.cursor()
    return cursor, conn
def get_encoder():
    cursor, conn = get_cursor()
    sql = 'select city, brand, output_volume from t_encoder order by id desc limit 1'
    cursor.execute(sql)
    result = cursor.fetchone()
    close(conn, cursor)
    return result

def get_statistics():
    cursor, conn = get_cursor()
    sql = 'select mean, var from t_statistics order by id desc limit 1'
    cursor.execute(sql)
    result = cursor.fetchone()
    close(conn, cursor)
    return result
def close(conn, cursor):
    cursor.close()
    conn.close()


def parse_data(req):
    data = req.get_data()
    data = json.loads(data)
    city = data.get('city')
    brand = data.get('brand')
    output_volume = data.get('output_volume')
    launch_year = data.get('launch_year')
    kilometres = data.get('kilometres')
    is_import = data.get('is_import')
    old_price = data.get('old_price')
    gear_type = data.get('gear_type')
    return city, brand, output_volume, launch_year, kilometres, is_import, old_price, gear_type

if __name__ == '__main__':
    app.run(debug=True) # 启动应用
