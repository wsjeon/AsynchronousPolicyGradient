# Learning CartPole using Asynchronous Reinforcement Learning
#
# The original code is in 
#
#   https://gym.openai.com/evaluations/eval_PqFieTTRCCwtOx96e8KMw
#
# where algorithm is implemented based on numpy. I try to implement 
# the same algorithm using in-graph replication and asynchronous learning 
# provided by TensorFlow.
import tensorflow as tf
import numpy as np
import gym
import time
from tensorflow.contrib import slim
from tensorflow.contrib.slim import fully_connected as fc

# Set hyperparameters.
flags = tf.app.flags
flags.DEFINE_float('GAMMA', 0.98, 'discount factor')
flags.DEFINE_float('LEARNING_RATE', 0.001, 'learning rate')
flags.DEFINE_integer('NUM_EPISODES', 2000, 'maximum episodes for training')
flags.DEFINE_boolean('MONITOR', False, 'monitor training frames or not')
flags.DEFINE_string('LOGDIR', './tmp', 'log directory')
flags.DEFINE_string('MONITORDIR', './monitor', 'directory for monitoring')
FLAGS = flags.FLAGS

# Environemt
env = gym.make('CartPole-v0')
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Neural network for policy approximation
observation_ = tf.placeholder(tf.float32, [None, observation_size])

def _net(net, hidden_layer_size=16):
  net = fc(net, hidden_layer_size, activation_fn=tf.nn.sigmoid, scope='fc0',
      weights_initializer =\
          tf.random_normal_initializer(stddev=1/np.sqrt(observation_size)))
  net = fc(net, action_size, activation_fn=tf.nn.softmax, scope='fc1', 
      weights_initializer =\
          tf.random_normal_initializer(stddev=1/np.sqrt(hidden_layer_size)))
  return net
policy = _net(observation_)

# Loss
action_ = tf.placeholder(tf.int32, [None])
advantage_ = tf.placeholder(tf.float32)

def _loss():
  log_policy = tf.log(policy)
  one_hot_action = tf.one_hot(action_, action_size)
  return -tf.reduce_sum(log_policy*one_hot_action) * advantage_
loss = _loss()

# Optimizer and train operator
optimizer = tf.train.GradientDescentOptimizer(FLAGS.LEARNING_RATE)
train_op = optimizer.minimize(loss)
  
# Summary
score_ = tf.placeholder(tf.float32)
tf.summary.scalar('score', score_)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(FLAGS.LOGDIR)

# Additional modules and operators for training
def act(observation):
  """Choose action based on observation and policy.

  Args:
    obs: Current observation.

  Returns:
    action: Action randomly chosen by using current policy.
  """
  current_policy = sess.run(policy, {observation_: [observation]})
  action = np.random.choice(action_size, p=current_policy[0])
  return action

def update(experience_buffer, returns):
  """Update neural network parameters based on eq.(20) in

    http://www.keck.ucsf.edu/~houde/sensorimotor_jc/possible_papers/JPeters08a.pdf

  In this code, the discounted sum of rewards given by the recent 100 episodes
  are used to generate the baseline. 

  Args:
    experience_buffer: all experiences in single episode.
    returns: list of discounted sums of rewards

  Returns:
    returns: list of discounted sums of rewards
  """
  rewards = np.array(experience_buffer[2])
  discount_rewards = rewards * (FLAGS.GAMMA ** np.arange(len(rewards)))
  current_return = discount_rewards.sum()
  returns.append(current_return)
  returns = returns[-100:] # Get recent 100 returns.
  baseline = sum(returns) / len(returns) # Baseline is the average of 100 returns.
  sess.run(train_op, {observation_: experience_buffer[0],
                      action_: experience_buffer[1],
                      advantage_: current_return - baseline}) 
  return returns

global_step = tf.get_variable('global_step', [],
    initializer = tf.constant_initializer(0),
    trainable = False,
    dtype = tf.int32)

counter_op = global_step.assign(global_step + 1)

# Training
with tf.Session() as sess:
  returns = [] # List to store returns of the last 100 episodes. 
  sess.run(tf.global_variables_initializer())

  # Monitor environment.
  if FLAGS.MONITOR:
    env.monitor.start(FLAGS.MONITORDIR, force = True)

  start_time = time.time() # To check learning time.

  # Training loop
  while True:

    # Intialization of episode. 
    timestep = 0; score = 0.0; experience_buffer = [[], [], []]
    episode = sess.run(global_step)
    observation = env.reset()

    # Agent-environment interaction
    while True:
      action = act(observation)
      experience_buffer[0].append(observation)
      experience_buffer[1].append(action)
      observation, reward, done, _ = env.step(action)
      experience_buffer[2].append(reward)
  
      timestep += 1; score += reward
      
      if done or timestep >= env.spec.timestep_limit:
        break
    
    # Update neural network.
    returns = update(experience_buffer, returns)
    sess.run(counter_op)

    # Log and tf summary.
    if episode % 10 == 0:
      print('episode: {0}\t|score: {1}\t|speed: {2} episodes/sec'.format(
        episode, score, (episode+1)/(time.time()-start_time)))
      summary_str = sess.run(summary_op, {score_: score})
      summary_writer.add_summary(summary_str, episode)
      summary_writer.flush()

    if episode + 1 == FLAGS.NUM_EPISODES:
      break
