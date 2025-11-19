# ChARGe Documentation

**ChARGe** is a **Ch**emical tool **A**ugmented **R**easoning models for **Ge**nerating molecules and reactions. 

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules Overview](#modules-overview)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

### Installation

```bash
pip install charge
```

or install from source with the GitHub repository:
```bash
pip install git+https://github.com/FLASK-LLNL/charge.git
```

For development, clone the repository and install with:
```bash
git clone https://github.com/FLASK-LLNL/charge.git
cd charge
pip install -e .
```

After installation, in order to install additional dependencies, run:
```bash
charge-install --extras all
```

### Quick Start

```python
from charge import some_function

result = some_function(...)
print(result)
```

### Modules Overview


- [`charge.clients`](reference/charge/clients): Client implementations for various backends.
- [`charge.servers`](reference/charge/servers): Server components.
- [`charge.tasks`](reference/charge/tasks): Task orchestration utilities.
- [`charge.inspector`](reference/charge/inspector): Inspection tools for debugging.
- [`charge.install`](reference/charge/install): Installation helpers.

### API Reference

Detailed API documentation can be found in the [Reference](reference/charge/index.md) section.

### Contributing

We welcome contributions! Please see the `CONTRIBUTORS` file on our [GitHub](https://github.com/FLASK-LLNL/charge) repository for guidelines.

### License

Please see the [LICENSE](LICENSE.md) in the documentation or the file in our repository [here](https://github.com/FLASK-LLNL/charge/blob/main/LICENSE) for details.