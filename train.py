import torch
from torch.utils.tensorboard import SummaryWriter
from food_classifier import model, train_loader, val_loader, test_loader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 导入 tqdm

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建SummaryWriter对象
writer = SummaryWriter()

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Model training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Training phase
    train_loss = 0.0
    model.train()  # Set the model to training mode
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation phase
    val_loss = 0.0
    model.eval()
    val_loader = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Record the training and validation loss for the current epoch
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 记录训练和验证损失到TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

# Test loop
test_loss = 0.0
model.eval()
test_loader = tqdm(test_loader, desc="Testing")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# 保存模型权重
torch.save(model.state_dict(), 'food_classifier.pth')

writer.close()  # 关闭SummaryWriter