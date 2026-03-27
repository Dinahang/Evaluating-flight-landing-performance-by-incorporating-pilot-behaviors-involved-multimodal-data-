# KEEP AT TOP! FIX FOR THE LOCALE ERROR
import os

# Force UTF-8 locale in environments where default locale can break parsing.
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

from pipeline_main import main  # type: ignore


if __name__ == '__main__':
    # Single entrypoint: delegates all work to the modular pipeline.
    main()
