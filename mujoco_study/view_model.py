import argparse
import mujoco
import mujoco.viewer

def load_model(file_path):
    """Load a model from the given file path."""
    return mujoco.MjModel.from_xml_path(file_path)

def create_data(model):
    """Create data for the given model."""
    return mujoco.MjData(model)

def view_model(model, data):
    """View the given model and data in a viewer."""
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)

def main():
    """Main function to load and view a model from a command line argument."""
    parser = argparse.ArgumentParser(description='Load and view a model.')
    parser.add_argument('file_path', help='Path to the model file')
    args = parser.parse_args()

    model = load_model(args.file_path)
    data = create_data(model)
    view_model(model, data)

if __name__ == "__main__":
    main()