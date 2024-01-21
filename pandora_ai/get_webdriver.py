import os
_root_=os.path.dirname(os.path.abspath(__file__))
import sys
if not sys.path[0]==_root_:
    sys.path.insert(0,_root_)
def root_join(*args):
    return os.path.join(_root_,*args)

from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from get_gecko_driver import GetGeckoDriver
from shutil import which

def get_webdriver():
    if not os.path.isdir(root_join("tmp")):
        os.mkdir(root_join("tmp"))
    os.environ['TMPDIR']=root_join("tmp")
    path=which('geckodriver')
    if not path:
        # Install the latest version of GeckoDriver
        get_driver = GetGeckoDriver()
        path=get_driver.install()
        print(f"geckodriver successfully installed to {path}")
    else:
        path=os.path.dirname(path)
    # Set the driver
    service=FirefoxService(log_path=os.path.devnull,executable_path=os.path.join(path,'geckodriver'))
    options = FirefoxOptions()
    # Enable headless mode
    options.add_argument('--headless')
    # Disable GPU (not necessary for headless)
    options.add_argument('--disable-gpu')
    # Set the window size (some websites require a minimum window size)
    options.add_argument('--window-size=1920,1080')
    # Instantiate a headless Firefox WebDriver
    driver = webdriver.Firefox(options=options, service=service)
    return driver