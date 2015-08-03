<?php
namespace library\dom\elements;
use library\dom\elements\element;
use library\dom\elements\components\text;
use library\dom\object;
use library\mediator\nodeElement;

/**
 * Description of paired
 *
 * @author durso
 */
abstract class paired extends element{

    public function addComponent(object $component){
        $this->node->addChild($component->getNode());
        if($this->isRendered){
            $this->updateJS('append',$component);
        }
    }
    public function removeComponent(object $component){
        $this->node->removeChild($component->getNode());
        if($component instanceof text){
            $this->updateJS('removeText');
        } else {
            $this->updateJS('remove',$component->getUid());
        }
    }
    /*
     * 
     * Render element to html
     * @return string
     */
    public function render(){
        $this->isRendered = true;
        $this->openTag();
        if($this->node->hasChild()){
            foreach($this->node->getChildren() as $child){
                $this->html .= $child->getValue()->render();
            }
        } 
        $this->closeTag();
        return $this->html;
    }
    
    protected function closeTag(){
        $this->html .= "</".$this->tag.">";
    }
    
    public function children($selector = false){
        return nodeElement::children($this,$selector);
    }
    public function find($selector){
        return nodeElement::find($this,$selector);
    }
    public function first($selector){
        return nodeElement::first($this,$selector);
    }

}
