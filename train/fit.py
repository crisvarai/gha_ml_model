import torch
import logging
import numpy as np

def fit(model, X, y, epochs, lr, device, weights_path):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = []
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        logging.info(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f}")
    torch.save(model.state_dict(), weights_path)
    logging.info("WEIGHTS-ARE-SAVED")