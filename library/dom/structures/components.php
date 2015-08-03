<?php

/**
 * Description of component
 *
 * @author durso
 */
namespace library\dom\structures;
use library\dom\object, library\dom\elements\components\elementFactory;


abstract class components{
    protected $root;
    protected $components = array();

    public function __construct($tag){
        $this->root = elementFactory::createByTag($tag);
    }
    public function addComponent(object $component){
        $this->root->addComponent($component);
        $this->tracker($component);
    }
    protected function tracker(object $component){
        if($component->hasTag()){
            $this->components[$component->getTag()][] = $component;
        }
    }
    public function getComponentByTagName($tag){
        if(isset($this->components[$tag])){
            return $this->components[$tag];
        }
    }
}

