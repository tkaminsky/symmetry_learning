import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
# Import torch dataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from controllers.oracle_controller import oracle_rev
from environment.symmetry_move import SymmetryMoveEnv
import wandb

wandb.init(project="symmetry-shift")


# Scale the image channels from 0-255 to -1 to 1

image_transforms = transforms.Compose(
    [
        # transforms.ToTensor(),
        # transforms.Resize((160, 160)),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

class ImageDataset(Dataset):
    def __init__(self, states, actions, transform=False):
        self.images = states
        self.actions = actions
        self.transform = transform
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.actions[idx]

        label = torch.tensor(label, dtype=torch.long)

        # Permute the image from (3,H,W) to (H,W, 3)
        # print("Before transform")
        # print(image.shape)
        image = torch.tensor(image)#.permute(1, 2, 0)
        image = image / 255 - .5

        image = image.float()

        # print(image.shape)

        # image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        # plt.imshow(image_np)
        # plt.title("In image, before transform")
        # plt.show()

        # image.permute(2, 0, 1)
        # print("After transform")
        # print(image.shape)

        # if self.transform is not None:
        #     image = self.transform(image).
        #     # print("After resize")
        #     # print(image.shape)

        # image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        # plt.imshow(image_np)
        # plt.title("In image, after transform")
        # plt.show()



        return image, label


# Create a CNN model
# Input: (3, 160, 160)
# Output: (6)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 40 * 40, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 6)

    def forward(self, x):
        # print("In forward pass")
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) 
        # print(x.shape)

        # View BATCHx16x40x40 as BATCHx16*40*40
        x = x.view(-1, 16 * 40 * 40)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # print(x.shape)
        return x

def get_trajectories(env, model, num_trajectories=100, max_steps=None):
    trajectories = []
    transform = transforms.Resize((160, 160))
    for i in range(num_trajectories):
        state, _ = env.reset()
        trajectory = {"states": [], "actions": []}
        if max_steps is None:
            max_steps = env.horizon
        for j in range(max_steps):
            # Turn state into tensor
            state = transform(torch.tensor(state)).numpy()

            action = oracle_rev(env)
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            state, _, done, _ = env.step(action)
            # state = state.numpy()
            # Parmute the state from (3, H, W) to (H, W, 3)
            # state = torch.tensor(state).permute(1, 2, 0).numpy()
            # plt.imshow(state)
            # plt.show()
            if done:
                break

        trajectories.append(trajectory)
    return trajectories

def traj_to_sa_pairs(trajectories):
    # Find state dimension
    dim = np.array(trajectories[0]["states"][0]).shape

    # Find number of elements
    num_objects = 0
    for trajectory in trajectories:
        num_objects += len(trajectory["states"])

    # Create empty numpy arrays for states and actions
    states = np.zeros((num_objects, dim[0], dim[1], dim[2]))
    actions = np.zeros((num_objects))
    counter = 0
    
    for i, trajectory in enumerate(trajectories):
        # Append the states and actions
        for j in range(len(trajectory["states"])):
            states[counter] = trajectory["states"][j]
            actions[counter] = trajectory["actions"][j]
            counter += 1

    return states, actions

def bc(env, model):
    # Number of trajectory samples to use for training/eval
    budget = 100
    eval_budget = 50

    losses = []
    accuracies = []
    test_losses = []
    test_accuracies = []
    # Number of times to go over the dataset with SGD
    num_epochs = 100
    batch_size = 64
    learning_rate = 1e-3

    # Number of stepf of GD to perform on each epoch
    NUM_ITERATIONS = 1000

    print("Gathering BC Training data . . .")

    # Get the trajectories
    trajectories = get_trajectories(env, model, num_trajectories=budget)

    states, actions = traj_to_sa_pairs(trajectories)

    state = states[0]
    # print(np.unique(state))
    # state = torch.tensor(state).permute(1, 2, 0).numpy().astype(np.uint8)

    # plt.imshow(state)
    # plt.show()

    print("Gathered BC Training data.")

    print("Gathering BC Test data . . .")

    # Turn to tensors
    # states = torch.from_numpy(states)
    # actions = torch.from_numpy(actions)

    test_trajectories = get_trajectories(env, model, num_trajectories=eval_budget)
    test_states, test_actions = traj_to_sa_pairs(test_trajectories)
    test_datset = ImageDataset(test_states, test_actions, transform=image_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_datset, batch_size=32, shuffle=True)

    print("Gathered BC Test data.")

    print("Dataset size: " + str(states.shape[0]) + " Test dataset size: " + str(test_states.shape[0]))
    
    dataset = ImageDataset(states, actions, transform=image_transforms)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create the loss function
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    print("Beginning BC Training . . .")

    iter_count = 0
    total_size = states.shape[0]
    total_test_size = test_states.shape[0]

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Train the model
    for epoch in range(num_epochs):

        # Shuffle dataset
        # dataloader.shuffle()
        loss_mean = 0
        accuracy_mean = 0
        count = 0
        for i, (states, actions) in enumerate(dataloader):
            if iter_count > NUM_ITERATIONS:
                break
            # Put the data on the GPU
            # if toggle:
            #     print("States shape: " + str(states.shape))
            #     im = states[0].permute(1, 2, 0).numpy()
            #     plt.imshow(im)
            #     plt.show()
            #     toggle = False
            states = states.cuda()
            # print(states.device)
            actions = actions.cuda()
            # print("State shape:")
            # print(states.shape)
            # 
            # Zero the gradients
            optimizer.zero_grad()

            # Get the output
            outputs = model(states)

            # print("Output shape:")
            # print(outputs.shape)

            # print("Actions shape:")
            # print(actions.shape)

            # Compute the loss
            loss = loss_fn(outputs, actions)

            # Backpropagate
            loss.backward()

            # Update the weights
            optimizer.step()

            loss_mean += loss.item() #* (len(actions) / total_size)

            # Compute the accuracy
            accuracy_mean += (torch.sum(torch.argmax(outputs, dim=1) == actions).item() / actions.shape[0])
            count += 1
            

            # Print the loss
            #print(f"Epoch {epoch}, Iteration {i}, Loss {loss.item()}")

        if iter_count > NUM_ITERATIONS:
            break


        # Compute the average loss (loss over num_batches)
        loss_mean /= count

        # # Compute the average accuracy
        accuracy_mean /= count

        # Print the loss
        print(f"Epoch {epoch}, Loss {loss_mean}, Accuracy {accuracy_mean}")
        # Log the loss and accuracy
        wandb.log({"loss": loss_mean, "accuracy": accuracy_mean})

        # Append the loss
        losses.append(loss_mean)

        # Append the accuracy
        accuracies.append(accuracy_mean)

        loss_mean = 0
        accuracy_mean = 0

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate the model
                # Set the model to evaluation mode

                count = 0

                for i, (states, actions) in enumerate(test_dataloader):
                    # Put the data on the GPU
                    states = states.cuda()
                    actions = actions.cuda()
                    # Get the output
                    outputs = model(states)

                    # Compute the loss
                    loss = loss_fn(outputs, actions)

                    loss_mean += loss.item() #* (len(actions) / total_test_size)

                    # Compute the accuracy
                    accuracy_mean += (torch.sum(torch.argmax(outputs, dim=1) == actions).item() / actions.shape[0])
                    count += 1

                # Compute the average loss
                loss_mean /= count

                # Compute the average accuracy
                accuracy_mean /= count

                # Print the loss
                print(f"Test Loss {loss_mean}, Test Accuracy {accuracy_mean}")
                # Log the loss and accuracy
                wandb.log({"test_loss": loss_mean, "test_accuracy": accuracy_mean})

                test_losses.append(loss_mean)
                test_accuracies.append(accuracy_mean)

            iter_count += 1

            model.train()


    # Save the model
    torch.save(model.state_dict(), "bc_model.pt")

    # Plot the loss and accuracy in two subplots
    fig, axs = plt.subplots(4)
    axs[0].plot(losses)
    axs[0].set_title("Loss")
    axs[1].plot(accuracies)
    axs[1].set_title("Accuracy")
    axs[2].plot(test_losses)
    axs[2].set_title("Test Loss")
    axs[3].plot(test_accuracies)
    axs[3].set_title("Test Accuracy")

    plt.suptitle(f"BC Loss | {budget} trajectories; {num_epochs} epochs; batch {batch_size}; learning rate {learning_rate}")
    plt.show()



def evaluate(env, model, num_trajectories=100, max_steps=None):
    # Get the trajectories
    trajectories = get_trajectories(env, model, num_trajectories=num_trajectories, max_steps=max_steps)

    # Compute the average reward
    rewards = []
    for trajectory in trajectories:
        rewards.append(np.sum(trajectory["rewards"]))
    return np.mean(rewards)

def bcRunner(env):
    model = CNN()
    # Make a resnet with an output of 6
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    # model.fc = nn.Linear(512, 6)

    # Put the model on the GPU
    model = model.cuda()
    bc(env, model)

    # Evaluate the model
    # avg_reward = evaluate(env, model)

    # print("Average reward attained: " + str(avg_reward))


    return model, 0 #, avg_reward