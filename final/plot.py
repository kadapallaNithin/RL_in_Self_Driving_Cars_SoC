import matplotlib.pyplot as plt

data = [[12.3,9.43],
[17.4,12.9],
[22.4,16.6],
[29.8,22],
[31.3,22.5],
[36.2,26.4],
[38,27.1],
[37.3,26.6],
[38.1,27.1],
[38.4,27.5],
[38.5,27.3],
[38.4,27.6],
[38.8,27.8],
[39.3,28.1],
[39.1,28],
[39.2,28.2],
[39.3,28.4],
[39.4,28.8],
[38,27.6],
[39.2,28.6],
[39,28.6],
[39.4,28.8],
[39.6,29.2],
[39.5,29.1],
[39.5,29.1],
[39.1,28.9]]

lengths = [x[0] for x in data]
rewards = [x[1] for x in data]

plt.figure()
plt.plot(rewards)
plt.title('Average Reward')
plt.xlabel('Training Rollouts')
plt.ylabel('Episode Reward Mean')
plt.savefig('avg_reward.png')

plt.figure()
plt.plot(lengths)
plt.title('Training Length')
plt.xlabel('Training Rollouts')
plt.ylabel('Episode Length Mean')
plt.savefig('training_length.png')