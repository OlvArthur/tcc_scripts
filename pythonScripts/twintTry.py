import json
import twint
import sys
print(sys.path)

# import re
# from optimus import Optimus

# op = Optimus(master="local")


def twintFunc():
    print('Start scrapping')
    c = twint.Config()
    c.Username = "3gerardpique"
    c.Lang = "pt"
    # c.Search = "great"
    # c.Format = "Tweet id: {id} | Tweet: {tweet} | Username: {username} | Date: {date}"
    c.Format = "Username: {username} | Tweet: {tweet}"
    # c.Limit = 1
    c.Pandas = True
    print(twint.output.panda.Tweets_df.columns)
    # twint.run.Search(c)


# twintFunc()
print('scrappe done')
# y = re.split(' |, ',txt)
# results = json.dumps(twint.run.Search(c), separators=(' '))
# print(results)
