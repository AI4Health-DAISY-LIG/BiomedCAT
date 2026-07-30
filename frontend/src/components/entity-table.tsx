import TypeBadge from "@/components/type-badge";
import { Cell, Row, Table } from "@/components/table";
import type { Entity } from "@/lib/runs";

export default function EntityTable({ entities }: { entities: Entity[] }) {
  return (
    // Capped so a long deck does not leave the rest of the row empty beside it.
    <Table headers={["Entity", "Type", "CURIE"]} maxHeight="22rem">
      {entities.map((entity, i) => (
        <Row key={`${entity.text}-${i}`}>
          <Cell>{entity.text}</Cell>
          <Cell>
            <TypeBadge type={entity.type} />
          </Cell>
          <Cell className="font-mono text-label">
            {entity.curie ?? <span className="text-muted">—</span>}
          </Cell>
        </Row>
      ))}
    </Table>
  );
}
