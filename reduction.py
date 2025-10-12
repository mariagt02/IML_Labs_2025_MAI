from red_techniques import Reductor
from dataset import DatasetLoader, DatasetVisualizer
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    dataset_loader = DatasetLoader(
        dataset_name="pen-based",
        dataset_dir="preprocessed"
    )
    dataset_loader.load()
    
    train_df, test_df = dataset_loader[0]
    train_df = train_df.to_numpy()
    
    visualizer = DatasetVisualizer(dataset_name=dataset_loader.dataset_name)
    _, _, pca = visualizer.visualize_df(num_dims=2)
    
    print(f"MCNN Before reduction: {len(train_df)}")
    train_reduced = Reductor.MCNN.reduce(train_df)
    print(f"MCNN After reduction: {len(train_reduced)}")
    
    x_train_pca = pca.transform(train_reduced[:,:-1])
    x_train_pca = pd.DataFrame(data=x_train_pca, columns=["principal component 1", "principal component 2"])
    plt.figure()
    plt.figure(figsize=(15,15))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title("Principal Component Analysis",fontsize=20)
    
    targets = set(train_reduced[:,-1])
    for target in targets:
        indicesToKeep = train_reduced[:,-1] == target
        plt.scatter(x_train_pca.loc[indicesToKeep, 'principal component 1']
                    , x_train_pca.loc[indicesToKeep, 'principal component 2'], s = 50)

    plt.legend(targets,prop={'size': 15}, loc='upper right')
    plt.show()
    # print(f"Allknn Before reduction: {len(train_df)}")
    # train_reduced = Reductor.ALLKNN.reduce(train_df, k=5)
    # print(f"Allknn After reduction: {len(train_reduced)}")
    
    
    # print(f"ICF Before reduction: {len(train_df)}")
    # train_reduced = Reductor.ICF.reduce(train_df, k = 3)
    # print(f"ICF After reduction: {len(train_reduced)}")
    
    

    