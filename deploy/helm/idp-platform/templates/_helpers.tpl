{{/*
Common labels for all resources.
*/}}
{{- define "idp.labels" -}}
app.kubernetes.io/part-of: idp-platform
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Agent-specific labels.
Usage: {{ include "idp.agentLabels" (dict "agentName" "router" "root" .) }}
*/}}
{{- define "idp.agentLabels" -}}
app.kubernetes.io/name: agent-{{ .agentName }}
app.kubernetes.io/component: agent
idp/agent-type: {{ .agentName }}
{{ include "idp.labels" .root }}
{{- end }}

{{/*
Full image path with optional registry prefix.
Usage: {{ include "idp.image" (dict "image" .Values.images.agent "global" .Values.global) }}
*/}}
{{- define "idp.image" -}}
{{- if .global.imageRegistry -}}
{{ .global.imageRegistry }}/{{ .image.repository }}:{{ .image.tag }}
{{- else -}}
{{ .image.repository }}:{{ .image.tag }}
{{- end -}}
{{- end }}

{{/*
Kafka topic name for an agent.
*/}}
{{- define "idp.kafkaTopic" -}}
agent.{{ . }}.requests
{{- end }}

{{/*
Kafka consumer group for an agent.
*/}}
{{- define "idp.consumerGroup" -}}
idp-agent-{{ . }}
{{- end }}
