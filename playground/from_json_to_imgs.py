import json
from PIL import Image

def get_data():
    f = open('data.json')
    data = json.load(f)
    f.close()
    return data


def filter(data, key, value):
    new = []
    for d in data:
        if key in d and d[key] == value:
            new.append(d)
    return new


def get_statistics(data, key):
    counter = dict()
    err = []
    for d in data:
        if not key in d:
            err.append(d)
            continue
        t = d[key]
        if not t in counter:
            counter[t] = 0
        counter[t] += 1
    return counter, err


def get_by_names(data, names):
    # data is normal, names is set which
    ret = []
    for d in data:
        n = d["Name"]
        if n in names:
            ret.append(d)
            names.remove(n)
    print("didnt find: ", names)
    return ret


def moves_to_lists(moves):
    start, mid, stop = [], [], []
    for m in moves:
        if m['IsStart']:
            start.append(m['Position'])
        elif m['IsEnd']:
            stop.append(m['Position'])
        else:
            mid.append(m['Position'])
    return start, mid, stop

# 7A+
def grade_transformer(grade):
    if "A" in grade:
        bonus = 0
    elif "B" in grade:
        bonus = 2
    elif "C" in grade:
        bonus = 4
    if "+" in grade:
        bonus += 1
    return int(grade[0]) * 6 + bonus - 37


def get_important_stuff(data):
    ret = dict()
    for d in data:
        grade = grade_transformer(d['Grade'])
        if not grade in ret:
            ret[grade] = []
        ret[grade].append(moves_to_lists(d['Moves']))
    return ret


def decode_coord(coord):
    char = coord[0]
    number = coord[1:]
    return ord(char)-65, -int(number)+18


def write_to_img(img, lis, color):
    for c in lis:
        x, y = decode_coord(c)
        img.putpixel((x, y), color)


data = get_data()
data = filter(data, "Method", "Feet follow hands")
data = filter(data, "MoonBoardHoldSetup", "MoonBoard Masters 2017")
data = filter(data, "MoonboardConfiguration", "40° MoonBoard")
# print(get_statistics(data, "MoonboardConfiguration")[0])
# exit(0)
data = get_important_stuff(data)

for grade, boulders in data.items():
    for i, b in enumerate(boulders):
        img = Image.new('RGB', (11, 18))
        s, m, e = b
        write_to_img(img, s, (0, 255, 0))
        write_to_img(img, m, (0, 0, 255))
        write_to_img(img, e, (255, 0, 0))
        img.save('out/'+str(grade)+'_'+str(i)+'.png')
