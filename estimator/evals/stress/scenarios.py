"""Three scripted 20-turn conversations with fact-trackers (block 2 of the exercise).

Each scenario is a list of ``ScenarioTurn`` sent IN ORDER to one session, so
turn N sees whatever the CAG kept of turns 1..N-1. Every turn declares the
fact it introduces (``fact`` + ``aliases``): the runner checks with
``MemoryDriftMetric`` whether later snapshots still contain it.

- ``growing``       — requirements pile up coherently (SSO, multi-tenant,
                      audit, CSV...). Measures the cost curve and whether the
                      turn-1 project name and the early facts survive turn 20.
- ``pivot``         — turn 5 swaps React Native for Flutter. Does the memory
                      update cleanly or does ``mentioned_technologies`` keep both?
- ``contradiction`` — turn 3 says budget 30.000 EUR, turn 8 says 80.000 EUR.
                      Which one survives, and where?

NOTA: con MAX_TURNS=6 la ventana lleva el turno en curso + los 5 anteriores.
A partir del turno 7 el turno 1 ya no viaja literal: solo sobrevive lo que el
extractor guardó en ``ProjectMetadata`` o lo que las respuestas repiten. Por
eso 20 turnos: más de 3 veces la ventana.

Los hechos se eligen para que NO aparezcan en el texto de los PDF de prueba
(``fixtures/build_pdfs.py``); un test lo comprueba. Si no, el adjunto
"recordaría" hechos que nadie dijo.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType

MIN_TURNS = 20


@dataclass(frozen=True)
class ScenarioTurn:
    """One scripted turn: what the client says and the fact later turns must remember."""

    transcript: str
    fact: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Supersession:
    """A fact that a later turn explicitly replaces (pivot / contradiction).

    From ``at_turn`` on, ``old`` should no longer be presented as current and
    ``new`` should be remembered.
    """

    at_turn: int
    old: str
    old_aliases: tuple[str, ...]
    new: str
    new_aliases: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    name: str
    project_type: ProjectType
    turns: list[ScenarioTurn]
    detail_level: DetailLevel = DetailLevel.MEDIUM
    output_format: OutputFormat = OutputFormat.PHASES_TABLE
    supersession: Supersession | None = None

    def __post_init__(self) -> None:
        if len(self.turns) < MIN_TURNS:
            raise ValueError(
                f"scenario {self.name!r} has {len(self.turns)} turns; at least "
                f"{MIN_TURNS} are needed to push the sliding window well past its cap"
            )
        if not self.turns[0].fact:
            raise ValueError("turn 1 must introduce the project name as its fact")


T = ScenarioTurn

# --- growing ------------------------------------------------------------------

_GROWING = [
    T(
        "Queremos un SaaS B2B llamado Nimbus para equipos de compras. Stack: React y "
        "PostgreSQL. Equipo de 3 personas. Alcance inicial: catálogo de proveedores, "
        "flujo de solicitudes de cotización y paneles básicos.",
        "Nimbus",
    ),
    T("Añade inicio de sesión único con Okta para empleados y proveedores externos.", "Okta"),
    T(
        "Presupuesto confirmado: 60000 EUR para la primera versión. El equipo sigue en 3.",
        "60000",
        ("60.000", "60,000", "60 000", "60k"),
    ),
    T(
        "Necesitamos un registro de auditoría de cada aprobación, guardado 7 años "
        "en almacenamiento WORM.",
        "WORM",
    ),
    T("Añade importación CSV de los datos maestros de proveedores, unas 50.000 filas.", "CSV"),
    T(
        "Multi-tenant: lo revenderemos a dos empresas hermanas, con aislamiento estricto "
        "de datos por cliente.",
        "tenant",
    ),
    T("Notificaciones por email y por webhook de Microsoft Teams.", "Teams"),
    T(
        "API pública de solo lectura para que los socios consulten disponibilidad, "
        "con OAuth2 client credentials.",
        "OAuth2",
        ("OAuth 2",),
    ),
    T("Módulo de informes con KPIs por comprador y un resumen semanal exportable.", "KPI"),
    T("Internacionalización: español, inglés y portugués; la moneda sigue al cliente.", "portugu"),
    T("Interfaz responsive para aprobadores en el móvil, sin app nativa todavía.", "responsive"),
    T(
        "Proceso nocturno de conciliación con el ERP SAP mediante ficheros IDoc.",
        "IDoc",
    ),
    T(
        "Doble factor de autenticación para quien apruebe más de 10.000 EUR.",
        "doble factor",
        ("2FA", "two-factor", "dos factores", "MFA"),
    ),
    T("Exportación masiva de pedidos a XML con el perfil UBL que nos pasó compras.", "UBL"),
    T("RGPD: flujo de derecho al olvido para los contactos de proveedores.", "olvido"),
    T(
        "Feature flag por cliente para un panel experimental de sugerencias con IA.",
        "feature flag",
    ),
    T(
        "Portal de autoservicio donde los proveedores actualizan sus datos bancarios y "
        "fiscales, con aprobación de un administrador.",
        "autoservicio",
        ("self-service",),
    ),
    T(
        "Rendimiento: carga de página por debajo de 2 s con 200 usuarios concurrentes.",
        "200 usuarios",
    ),
    T("Recuperación ante desastres: RPO de 1 hora y RTO de 4 horas.", "RPO"),
    T("Último pedido: integración con DocuSign para la firma del contrato.", "DocuSign"),
]

# --- pivot ----------------------------------------------------------------------

_PIVOT = [
    T(
        "Construye una app para clientes llamada Helios para una cadena de tiendas. "
        "Stack: React Native para iOS y Android. Equipo de 4 personas.",
        "Helios",
    ),
    T(
        "Sustituye la tarjeta de fidelización: el socio escanea un QR en caja, acumula "
        "puntos y los canjea en la app.",
        "QR",
    ),
    T("Backend: una API en Node.js sobre la base de datos de clientes existente.", "Node"),
    T("Notificaciones push de ofertas segmentadas por región de tienda.", "push"),
    T(
        "CAMBIO DE STACK: el cliente estandariza todas sus apps en Flutter. Pasamos a "
        "Flutter y React Native queda descartado.",
        "Flutter",
    ),
    T("Confirmado Flutter para iOS y Android; el alcance no cambia.", "Flutter"),
    T("Añade pases de Apple Wallet y Google Wallet para la tarjeta.", "Wallet"),
    T("Integración con la API REST del TPV para acumular puntos.", "TPV"),
    T("Escaneo de código de barras de productos para ver su ficha.", "barras"),
    T("Ofertas por geocerca: una push cuando el socio está cerca de una tienda.", "geoc"),
    T("Login social con Apple ID y Google.", "Apple ID"),
    T("Panel web interno para que marketing programe campañas, hecho en Vue.js.", "Vue"),
    T("Analítica: eventos a Segment y de ahí a BigQuery.", "BigQuery"),
    T("Chat de soporte dentro de la app integrado con Zendesk.", "Zendesk"),
    T("Tickets digitales que sobreviven aunque falle la impresora de caja.", "ticket"),
    T("Idiomas: castellano y catalán; sin inglés en la v1.", "catal"),
    T("Informes de fallos y trazas de rendimiento con Firebase Crashlytics.", "Crashlytics"),
    T("Ficha de tienda (ASO) y capturas para 3 tamaños de pantalla.", "ASO"),
    T("Cumplimiento: flujo de consentimiento RGPD en la primera apertura.", "consentimiento"),
    T(
        "Último pedido: modo sin conexión para ver la tarjeta y los últimos premios.",
        "sin conexi",
        ("offline",),
    ),
]

# --- contradiction ----------------------------------------------------------------

_CONTRADICTION = [
    T(
        "Construye una herramienta interna llamada Atlas para el departamento financiero: "
        "flujo de aprobación de facturas de proveedores. Stack: Rails y PostgreSQL.",
        "Atlas",
    ),
    T("El equipo son 2 desarrolladores y un diseñador a media jornada.", "media jornada"),
    T(
        "Tope de presupuesto: 30000 EUR para todo el proyecto.",
        "30000",
        ("30.000", "30,000", "30 000", "30k"),
    ),
    T("Registro de quién aprobó qué, exportable a Excel.", "Excel"),
    T("Todos los aprobadores entran con el SSO corporativo de Azure AD.", "Azure"),
    T("Panel de aprobaciones pendientes agrupadas por aprobador.", "pendientes"),
    T("Aprobación masiva de hasta 50 facturas con un paso de confirmación.", "50 facturas"),
    T(
        "CONTRADICCIÓN: dirección ha subido el presupuesto. El nuevo es 80000 EUR; la "
        "cifra de 30000 queda obsoleta. Planifica con el nuevo.",
        "80000",
        ("80.000", "80,000", "80 000", "80k"),
    ),
    T(
        "Confirmado el nuevo presupuesto de 80000 EUR; los 30k eran provisionales.",
        "80000",
        ("80.000", "80,000", "80 000", "80k"),
    ),
    T("Conciliación a tres bandas contra pedidos y albaranes.", "albaran", ("albarán",)),
    T("Integración con el ERP Oracle EBS por su pasarela REST.", "Oracle"),
    T("Enrutado de cada factura al aprobador de su centro de coste.", "centro de coste"),
    T("Correo diario con el resumen de pendientes de cada aprobador.", "correo diario"),
    T("Captura asistida por OCR de facturas en papel con AWS Textract.", "Textract"),
    T("Facturas multidivisa normalizadas con el cambio diario del BCE.", "BCE", ("multidivisa",)),
    T("Informes de antigüedad de deuda y exposición total por proveedor.", "antig"),
    T("Hilo de comentarios en cada factura para hablar entre aprobadores.", "comentarios"),
    T("Rendimiento: 5000 facturas al día en el pico de cierre de mes.", "5000", ("5.000",)),
    T("Principio de cuatro ojos para facturas de más de 25.000 EUR.", "cuatro ojos"),
    T("Último pedido: exportación nocturna de facturas aprobadas a Snowflake.", "Snowflake"),
]

_SCENARIOS: dict[str, Scenario] = {
    "growing": Scenario("growing", ProjectType.WEB_SAAS, _GROWING),
    "pivot": Scenario(
        "pivot",
        ProjectType.MOBILE_APP,
        _PIVOT,
        supersession=Supersession(5, "React Native", (), "Flutter", ()),
    ),
    "contradiction": Scenario(
        "contradiction",
        ProjectType.INTERNAL_TOOL,
        _CONTRADICTION,
        supersession=Supersession(
            8,
            "30000",
            ("30.000", "30,000", "30 000", "30k"),
            "80000",
            ("80.000", "80,000", "80 000", "80k"),
        ),
    ),
}


def get_scenario(name: str) -> Scenario:
    try:
        return _SCENARIOS[name]
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; available: {sorted(_SCENARIOS)}") from None


def all_scenarios() -> list[Scenario]:
    return list(_SCENARIOS.values())
