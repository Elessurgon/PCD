import pandas as pd
import datetime
import os


def add(pin):
    ct = datetime.datetime.now()
    time_pin = pd.DataFrame({"current time": [], "pin": []})

    if os.path.isfile('./pins/pins.csv') and os.path.isfile('./pins/frequency.csv'):
        time_pin = pd.read_csv('./pins/pins.csv')
    else:
        print("CREATED")
        time_pin = pd.DataFrame({"Time": [], "Pin code": []})

    df = pd.DataFrame.from_dict({"Time": [ct], "Pin code": [pin]})

    time_pin = time_pin.append(df)
    freq = {}
    for r in time_pin.iterrows():
        v = r[1]

        if int(v[1]) in freq:
            freq[int(v[1])] += 1
        else:
            freq[int(v[1])] = 1

    freq = pd.DataFrame(freq.items(), columns=['Pin code', 'Count'])
    time_pin.to_csv('./pins/pins.csv', index=False)
    freq.to_csv('./pins/frequency.csv', index=False)


if __name__ == '__main__':
    add('10')
    add('20')
    add('10')
