import os, socket
from earhart import __path__

DATADIR = os.path.join(os.path.dirname(__path__[0]), 'data')
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
PHOTDIR = os.path.join(DATADIR, 'photometry')

from cdips import __path__ as cdipspath
ALLVARDATADIR = os.path.join(os.path.dirname(cdipspath[0]), 'results',
                             'allvariability_reports', 'NGC_2516', 'data')

LOCALDIR = os.path.join(os.path.expanduser('~'), 'local', 'earhart')
if not os.path.exists(LOCALDIR):
    os.mkdir(LOCALDIR)
