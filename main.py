# Learning CartPole using Asynchronous Reinforcement Learning
#
# The original code is in 
#
#   https://gym.openai.com/evaluations/eval_PqFieTTRCCwtOx96e8KMw
#
# where algorithm is implemented based on numpy.
# 
# This code is implemented by using Distributed TensorFlow.
#
# Wonseok Jeon at KAIST
# wonsjeon@kaist.ac.kr
import tensorflow as tf
import numpy as np
import gym
import time
from multiprocessing import Process
from tensorflow.contrib import slim
from tensorflow.contrib.slim import fully_connected as fc

# Set hyperparameters.
flags = tf.app.flags
flags.DEFINE_float('GAMMA', 0.98, 'discount factor')
flags.DEFINE_float('LEARNING_RATE', 0.001, 'learning rate')
flags.DEFINE_integer('NUM_EPISODES', 400, 'maximum episodes for training')
flags.DEFINE_string('LOGDIR', './tmp', 'log directory')
flags.DEFINE_string('job_name', 'worker', 'job name: worker or ps (parameter server)')
flags.DEFINE_integer('NUM_WORKERS', 1, 'number of workers')
flags.DEFINE_integer('task_index', 0, 'task index of server')
FLAGS = flags.FLAGS

# Cluster
worker = ['localhost:2220']
ps = ['localhost:{}'.format(2221+i) for i in range(FLAGS.NUM_WORKERS)]
cluster = tf.train.ClusterSpec({
  'worker': worker,
  'ps': ps})

# Environment: Each task will run environments indepedently.
env = gym.make('CartPole-v0')
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Neural network for policy approximation
def _net(net, hidden_layer_size=16):
  net = fc(net, hidden_layer_size, activation_fn=tf.nn.sigmoid, scope='fc0',
      weights_initializer =\
          tf.random_normal_initializer(stddev=1/np.sqrt(observation_size)))
  net = fc(net, action_size, activation_fn=tf.nn.softmax, scope='fc1', 
      weights_initializer =\
          tf.random_normal_initializer(stddev=1/np.sqrt(hidden_layer_size)))
  return net

# Shared memory
with tf.variable_scope('shared_memory'):
  """Global shared memory
  Note: The scope 'shared_memory' is to share the followings:
    1) parameters
    2) step counter
    3) optimizer
    4) summary
  """
  with tf.device('/job:ps/task:0/cpu:0'):

    # Global shared parameters 
    observation_ = tf.placeholder(tf.float32, [None, observation_size], name='observation')
    policy = _net(observation_)
    global_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_memory')

    # Global step counter
    global_step_counter = tf.get_variable('global_step_counter', [],
        initializer = tf.constant_initializer(),
        trainable = False,
        dtype = tf.int32)

    # Global shared optimizer ???
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.LEARNING_RATE)

    # Summary
    score_ = tf.placeholder(tf.float32, name='score_')
    tf.summary.scalar('score', score_)
    summary_op = tf.summary.merge_all()
    
    # Global counter
    counter_op = global_step_counter.assign(global_step_counter + 1)

# Workers
for i in range(FLAGS.NUM_WORKERS):
  scope = 'worker%d'%i
  with tf.variable_scope(scope):
    with tf.device('/job:worker/task:%d/gpu:0'%i):
  
      # Local network and parameters
      observation_ = tf.placeholder(tf.float32, [None, observation_size], name='observation')
      policy = _net(observation_)
      local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  
      # Assign operator: global shared parameters -> local parameters
      assign_op = tf.group(*[
        local_param.assign(global_param)
        for local_param, global_param in zip(local_params, global_params)],
        name='synchronizer')  
  
      # Loss
      action_ = tf.placeholder(tf.int32, [None], name='action')
      advantage_ = tf.placeholder(tf.float32, name='advantage')
  
      def _loss():
        log_policy = tf.log(policy)
        one_hot_action = tf.one_hot(action_, action_size)
        return -tf.multiply(
            tf.reduce_sum(log_policy*one_hot_action),
            advantage_,
            name='loss')
      loss = _loss()
  
      # Gradients w.r.t. local parameters
      grads = tf.gradients(loss, local_params)
  
      # Apply gradients to global shared parameters 
      grads_and_vars = zip(grads, global_params)
      train_op = optimizer.apply_gradients(grads_and_vars, name='grad_applier')
       
# Additional modules and operators
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

# Define task.
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == 'ps':
  env.close()
  server.join() # See the issue (https://github.com/tensorflow/tensorflow/issues/4713).

elif FLAGS.job_name == 'worker':
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.3
  with tf.Session(server.target, config=config) as sess:

    # Summary writer
    summary_writer = tf.summary.FileWriter(FLAGS.LOGDIR, sess.graph)
    
    # Variable initialization
    sess.run(tf.global_variables_initializer())

    # List to stre returns of the last 100 episodes
    returns = []

    # To check learning speed
    start_time = time.time()
  
    # Training loop
    while True:
      
      # Global shared parameter -> local parameters
      sess.run(assign_op)
  
      # Intialization of episode. 
      timestep = 0; score = 0.0; experience_buffer = [[], [], []]
      episode = sess.run(global_step_counter)
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
        print('worker{0}\t|episode: {1}\t|score: {2}\t|speed: {3} episodes/sec'.format(
          FLAGS.task_index, episode, score, 10/(time.time()-start_time)))
        summary_str = sess.run(summary_op, {score_: score})
        summary_writer.add_summary(summary_str, episode)
        summary_writer.flush()
        start_time = time.time()
  
      if episode + 1 == FLAGS.NUM_EPISODES:
        break

  env.close()
