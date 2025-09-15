"""
# --8<-- [start:execute]
result = pdsh_q3(customer, lineitem, orders).remote(ctx).distributed().execute()
# --8<-- [end:execute]

# --8<-- [start:await_progress]
result.await_progress().data
# --8<-- [end:await_progress]

# --8<-- [start:await_summary]
result.await_progress().summary
# --8<-- [end:await_summary]
"""
