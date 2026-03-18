import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

df = pd.read_csv("house_price_prediction_.csv")
df = df.drop("id", axis=1)

cat_col = df.select_dtypes(include="object")
for col in cat_col:
    df[col] = LabelEncoder().fit_transform(df[col])

x = df.drop("price", axis=1)
y = df["price"]

x_scaler = StandardScaler()
y_scaler = StandardScaler()
X = x_scaler.fit_transform(x)
Y = y_scaler.fit_transform(y.values.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class HOUSE_PRICE_NN(nn.Module):
    def __init__(self, input_dim):
        super(HOUSE_PRICE_NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = HOUSE_PRICE_NN(x_train.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    y_pred_orig = y_scaler.inverse_transform(y_pred.numpy())
    y_test_orig = y_scaler.inverse_transform(y_test.numpy())

accuracy = r2_score(y_test_orig, y_pred_orig) * 100
print(f"Neural Network R² Score: {accuracy:.2f}%")