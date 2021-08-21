from statsmodels.iolib import SimpleTable

class _TableMakerMixin:

    def _single_summary(self, data, title=None):
        if not isinstance(data, dict):
            raise ValueError(f"`data` arg ({type(data)}) must be type `dict`")
        rownames, data = zip(*data.items())
        rownames = [rowname + ':' for rowname in rownames]
        return SimpleTable(data, None, rownames, title)

    def _twocol_summary(self, lhs_data, rhs_data, title=None):
        if not isinstance(lhs_data, dict) or not isinstance(rhs_data, dict):
            raise ValueError(f"Args must be type `dict`")

        lhs_rownames, lhs_data = zip(*lhs_data.items())
        lhs_rownames = [rowname + ':' for rowname in lhs_rownames]

        rhs_rownames, rhs_data = zip(*rhs_data.items())
        rhs_rownames = [rowname + ':' for rowname in rhs_rownames]

        lhs_table = SimpleTable(lhs_data, None, lhs_rownames, title)
        rhs_table = SimpleTable(rhs_data, None, rhs_rownames, title)

        lhs_table.extend_right(rhs_table)

        return lhs_table

    def summary(self, summary_type='basic'):
        """ Return Summary object """

        lhs_data, title = self._make_main_table_elements()
        rhs_data, _ = self._make_content_table_elements()

        if summary_type == 'basic':
            return self._twocol_summary(lhs_data, rhs_data, title)
        else:
            raise ValueError(summary_type)

