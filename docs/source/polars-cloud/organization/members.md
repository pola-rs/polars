# Manage members

The Members page allows you to manage who has access to the organization and see open invitations.
You can invite new users by email/url and select any workspaces they will automatically join once
they've accepted.

## Roles

There are two types of roles:

1. `Admin` - Able to do anything a member can and additionally invite new members, change roles or
   organization settings
2. `Member` - Able to see the organization and create new workspaces in it

There is also the notion of implicit and explicit roles. An organization admin is implicitly also a
workspace admin even though they are not an explicit member of the workspace.

Organization admins are able to promote others to admin or demote other admins to member. The system
enforces that there is always at least one organization admin. In case you are unable to contact the
only organization admin, please contact Polars Cloud support at
[support@polars.tech](mailto:support@polars.tech).
