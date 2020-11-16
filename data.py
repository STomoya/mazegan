
import glob

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

class MazeDataset(Dataset):
    def __init__(self,
        image_size, transform=None
    ):
        self.mazes  = sorted(glob.glob('./image/maze/*.jpg'))
        self.solved = sorted(glob.glob('./image/solved/*.jpg'))
        self.length = len(self.mazes)

        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ])

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        maze, solved = self.mazes[index], self.solved[index]
        maze, solved = Image.open(maze).convert('L'), Image.open(solved).convert('L')
        maze, solved = self.transform(maze), self.transform(solved)
        return maze, solved

if __name__ == "__main__":
    data = MazeDataset(128)
    for maze, solved in data:
        print(maze.size(), solved.size())
        break
