import pytest
@pytest.mark.parametrize('x',[0])
def import_modules(x):
    import maxplotlib
    import matplotlib

if __name__ == '__main__':
    import_modules(x=1)