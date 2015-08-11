import os
import ConfigParser

PWD = os.path.dirname(__file__)

config = ConfigParser.ConfigParser()
config.read(os.path.join(PWD, '..', '..', 'stroke.cfg'))

