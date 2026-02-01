"""
# --8<-- [start:execute]
result = pdsh_q3(customer, lineitem, orders).remote(ctx).distributed().execute()
# --8<-- [end:execute]

# --8<-- [start:await_profile]
result.await_profile().data
# --8<-- [end:await_profile]

# --8<-- [start:await_summary]
result.await_profile().summary
# --8<-- [end:await_summary]
"""
