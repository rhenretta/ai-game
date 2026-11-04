export type SpecialistPod = 'all' | 'core' | 'experience' | 'world' | 'engineering' | 'producer';

export interface RoleMetadata {
  role: string;
  label: string;
  shortLabel: string;
  pod: Exclude<SpecialistPod, 'all'>;
  accentVar: string;
}

const CANONICAL_ROLES: Record<string, RoleMetadata> = {
  CoProducer: {
    role: 'CoProducer',
    label: 'Co-Producer (Council Chair)',
    shortLabel: 'Co-Producer',
    pod: 'core',
    accentVar: 'var(--accent)',
  },
  CreativeDirector: {
    role: 'CreativeDirector',
    label: 'Creative Director',
    shortLabel: 'Director',
    pod: 'core',
    accentVar: 'var(--accent)',
  },
  GameplayDesigner: {
    role: 'GameplayDesigner',
    label: 'Gameplay Designer',
    shortLabel: 'Gameplay',
    pod: 'core',
    accentVar: 'var(--ok)',
  },
  LevelAndProgressionDesigner: {
    role: 'LevelAndProgressionDesigner',
    label: 'Level & Progression Designer',
    shortLabel: 'Level/Prog',
    pod: 'experience',
    accentVar: 'var(--ok)',
  },
  UXExperienceDesigner: {
    role: 'UXExperienceDesigner',
    label: 'UX & Experience Designer',
    shortLabel: 'UX',
    pod: 'experience',
    accentVar: 'var(--warn)',
  },
  StoryWorldArchitect: {
    role: 'StoryWorldArchitect',
    label: 'Story & World Architect',
    shortLabel: 'World',
    pod: 'world',
    accentVar: 'var(--purple)',
  },
  CharacterDesigner: {
    role: 'CharacterDesigner',
    label: 'Character Designer',
    shortLabel: 'Character',
    pod: 'world',
    accentVar: 'var(--purple)',
  },
  VisualArtDirector: {
    role: 'VisualArtDirector',
    label: 'Visual Art Director',
    shortLabel: 'Visual Art',
    pod: 'world',
    accentVar: 'var(--accent)',
  },
  AudioAtmosphereDesigner: {
    role: 'AudioAtmosphereDesigner',
    label: 'Audio & Atmosphere Designer',
    shortLabel: 'Audio',
    pod: 'world',
    accentVar: 'var(--warn)',
  },
  GameplaySystemsEngineer: {
    role: 'GameplaySystemsEngineer',
    label: 'Gameplay Systems Engineer',
    shortLabel: 'Systems SWE',
    pod: 'engineering',
    accentVar: 'var(--accent)',
  },
  QAVerificationLead: {
    role: 'QAVerificationLead',
    label: 'QA Verification Lead',
    shortLabel: 'QA Lead',
    pod: 'engineering',
    accentVar: 'var(--ok)',
  },
  ExecutiveProducer: {
    role: 'ExecutiveProducer',
    label: 'Executive Producer (You)',
    shortLabel: 'Producer',
    pod: 'producer',
    accentVar: 'var(--warn)',
  },
  executive_producer: {
    role: 'executive_producer',
    label: 'Executive Producer',
    shortLabel: 'Producer',
    pod: 'producer',
    accentVar: 'var(--warn)',
  },
  council: {
    role: 'council',
    label: 'Council Shared Memory',
    shortLabel: 'Council',
    pod: 'core',
    accentVar: 'var(--accent)',
  },
  ScriptwriterAgent: {
    role: 'ScriptwriterAgent',
    label: 'Scriptwriter',
    shortLabel: 'Script',
    pod: 'world',
    accentVar: 'var(--purple)',
  },
  task_lead: {
    role: 'task_lead',
    label: 'Task Lead Strategist',
    shortLabel: 'Task Lead',
    pod: 'engineering',
    accentVar: 'var(--accent)',
  },
  swe_engineer: {
    role: 'swe_engineer',
    label: 'Lead Gameplay SWE',
    shortLabel: 'SWE',
    pod: 'engineering',
    accentVar: 'var(--accent)',
  },
  qa_test_engineer: {
    role: 'qa_test_engineer',
    label: 'QA Test Engineer',
    shortLabel: 'QA',
    pod: 'engineering',
    accentVar: 'var(--ok)',
  },
  visual_artist: {
    role: 'visual_artist',
    label: 'Visual Artist',
    shortLabel: 'Visual',
    pod: 'world',
    accentVar: 'var(--purple)',
  },
  audio_designer: {
    role: 'audio_designer',
    label: 'Audio Designer',
    shortLabel: 'Audio',
    pod: 'world',
    accentVar: 'var(--warn)',
  },
  script_writer: {
    role: 'script_writer',
    label: 'Narrative Script Writer',
    shortLabel: 'Narrative',
    pod: 'world',
    accentVar: 'var(--purple)',
  },
  LeadGameplaySWEAgent: {
    role: 'LeadGameplaySWEAgent',
    label: 'Lead Gameplay SWE',
    shortLabel: 'SWE',
    pod: 'engineering',
    accentVar: 'var(--accent)',
  },
  QAPlaytesterBuildSupervisor: {
    role: 'QAPlaytesterBuildSupervisor',
    label: 'QA Build Supervisor',
    shortLabel: 'QA Supervisor',
    pod: 'engineering',
    accentVar: 'var(--ok)',
  },
  SystemsArchitectAgent: {
    role: 'SystemsArchitectAgent',
    label: 'Systems Architect',
    shortLabel: 'Architect',
    pod: 'engineering',
    accentVar: 'var(--accent)',
  },
  VisualArtistAgent: {
    role: 'VisualArtistAgent',
    label: 'Visual Artist',
    shortLabel: 'Visual',
    pod: 'world',
    accentVar: 'var(--purple)',
  },
  AudioDesignerAgent: {
    role: 'AudioDesignerAgent',
    label: 'Audio Designer',
    shortLabel: 'Audio',
    pod: 'world',
    accentVar: 'var(--warn)',
  },
  BlindPlaytesterAgent: {
    role: 'BlindPlaytesterAgent',
    label: 'Blind Playtester',
    shortLabel: 'Playtester',
    pod: 'experience',
    accentVar: 'var(--ok)',
  },
  StudioOrchestrator: {
    role: 'StudioOrchestrator',
    label: 'Studio Orchestrator',
    shortLabel: 'Orchestrator',
    pod: 'core',
    accentVar: 'var(--accent)',
  },
};

export const POD_OPTIONS: Array<{ id: SpecialistPod; label: string }> = [
  { id: 'all', label: 'All Pods' },
  { id: 'core', label: 'Core Vision' },
  { id: 'experience', label: 'Level & UX' },
  { id: 'world', label: 'World, Art & Audio' },
  { id: 'engineering', label: 'Engineering & QA' },
  { id: 'producer', label: 'Producer Steering' },
];

export function getRoleMetadata(role: string | undefined | null, displayName?: string): RoleMetadata {
  const key = (role ?? '').trim();
  if (key && CANONICAL_ROLES[key]) {
    const known = CANONICAL_ROLES[key];
    return displayName ? { ...known, label: displayName } : known;
  }
  return {
    role: key || 'Specialist',
    label: displayName || key || 'Studio Specialist',
    shortLabel: key || 'Agent',
    pod: 'engineering',
    accentVar: 'var(--text-secondary)',
  };
}
