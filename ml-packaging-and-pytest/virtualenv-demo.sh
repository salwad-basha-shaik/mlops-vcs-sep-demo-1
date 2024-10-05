pip install virtualenv

# Step 2: Create two virtual environments
virtualenv demo1
virtualenv demo2

# Step 3: Install different versions of Pandas
demo1/bin/pip install pandas==2.2.0
demo2/bin/pip install pandas==2.1.2

# Step 4: Verify the installed versions
demo1/bin/python -c "import pandas as pd; print('Pandas version in env1:', pd.__version__)"
demo2/bin/python -c "import pandas as pd; print('Pandas version in env2:', pd.__version__)"