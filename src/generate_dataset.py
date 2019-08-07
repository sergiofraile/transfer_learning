import sys
import os
import zipfile

local_zip = str(sys.argv[1])

zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./datasets')
zip_ref.close()
