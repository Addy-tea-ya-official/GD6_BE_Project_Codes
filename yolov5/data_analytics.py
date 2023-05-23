import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/output.csv')


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
df.plot.bar(x="Enhancement method", y="Precision", rot=70, color = "red", title="Precision for different models", ax=axes[0,0])
df.plot.bar(x="Enhancement method", y="Recall", rot=70,color = "purple", title="Recall for different models", ax=axes[0,1])
df.plot.bar(x="Enhancement method", y="mAP50", rot=70, color = "green",title="mAP50 for different models", ax=axes[1,0])
df.plot.bar(x="Enhancement method", y="mAP50-95", rot=70, title="mAP50-95 for different models", ax=axes[1,1])
plt.subplots_adjust(left=0.1,
                    bottom=0.2,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.8)
axes[0][0].legend(loc='upper right')
axes[0][1].legend(loc='upper right')
axes[1][0].legend(loc='upper right')
axes[1][1].legend(loc='upper right')
plt.savefig('output/analytics.png')
plt.show(block=True)

df.plot.bar(x="Enhancement method", y="fitness", rot=70, title="fitness for different models")
plt.savefig('output/fitness.png')
plt.show(block=True)
