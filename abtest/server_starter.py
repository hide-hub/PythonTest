# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from flask import Flask, jsonify, request
from scipy.stats import beta

# create an app
app = Flask(__name__)


# define bandits
# there's no "pull arm" here
# since that's technically now the user/client
class Bandit:
  def __init__(self, name):
    self.name = name
    self.count_clicked = 1
    self.count_show = 0

  def sample(self):
    count_notclicked = self.count_show - self.count_clicked + 2
    return np.random.beta( self.count_clicked, count_notclicked )

  def update(self, action):
    self.count_clicked += action
  
  def showen(self):
    self.count_show += 1


# initialize bandits
banditA = Bandit('A')
banditB = Bandit('B')



@app.route('/get_ad')
def get_ad():
  sampleA = banditA.sample()
  sampleB = banditB.sample()
  if ( sampleA > sampleB ):
    adtype = 'A'
    banditA.showen()
  else:
    adtype = 'B'
    banditB.showen()
  return jsonify({'advertisement_id': adtype})


@app.route('/click_ad', methods=['POST'])
def click_ad():
  result = 'OK'
  if request.form['advertisement_id'] == 'A':
    banditA.update( 1 )
    pass
  elif request.form['advertisement_id'] == 'B':
    banditB.update( 1 )
    pass
  else:
    result = 'Invalid Input.'

  # nothing to return really
  return jsonify({'result': result})


if __name__ == '__main__':
  app.run(host='127.0.0.1', port='8888')